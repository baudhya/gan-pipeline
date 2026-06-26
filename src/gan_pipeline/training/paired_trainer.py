from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.models.losses import LossType
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.training.base_trainer import BaseTrainer


def _make_lr_lambda(n_epochs_keep: int, n_epochs_decay: int) -> Callable[[int], float]:
    """Linear LR schedule: constant for n_epochs_keep, then decays to 0."""

    def _lambda(epoch: int) -> float:
        return 1.0 - max(0, epoch - n_epochs_keep) / float(n_epochs_decay + 1)

    return _lambda


class PairedGANTrainer(BaseTrainer):
    """Abstract trainer for conditional paired SAR→EO translation.

    Handles scaffolding shared by all paired conditional GAN variants:
    multi-scale discriminator alias, linear LR decay, fixed sample capture,
    SAR / fake / real visualisation, and D + G step orchestration.

    Subclasses must implement: _log_params, _d_step, _g_step.
    """

    _ms_disc: MultiScaleDiscriminator

    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: MultiScaleDiscriminator,
        cfg: DictConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        super().__init__(generator, discriminator, cfg, device, output_dir)
        self._ms_disc = discriminator
        self.loss_type = LossType(cfg.training.loss_type)
        self.label_smoothing: float = float(cfg.training.get("label_smoothing", 1.0))
        self.fixed_sar: torch.Tensor | None = None
        self.fixed_eo: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Scheduler hooks
    # ------------------------------------------------------------------

    def _build_schedulers(self) -> None:
        n_decay = self.cfg.training.epochs // 2
        n_keep = self.cfg.training.epochs - n_decay
        self._lr_lambda = _make_lr_lambda(n_keep, n_decay)
        self.sched_g = LambdaLR(self.opt_g, self._lr_lambda)
        self.sched_d = LambdaLR(self.opt_d, self._lr_lambda)

    def _step_schedulers(self) -> None:
        self.sched_g.step()
        self.sched_d.step()

    def _restore_schedulers(self, start_epoch: int) -> None:
        self.sched_g = LambdaLR(self.opt_g, self._lr_lambda, last_epoch=start_epoch - 1)
        self.sched_d = LambdaLR(self.opt_d, self._lr_lambda, last_epoch=start_epoch - 1)

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def _step_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sar: torch.Tensor = batch["sar"]
        eo: torch.Tensor = batch["eo"]
        if self.fixed_sar is None:
            self.fixed_sar = sar[:8].to(self.device)
            self.fixed_eo = eo[:8].to(self.device)
        return self._train_step(sar, eo)

    def _save_samples(self, epoch: int) -> None:
        assert self.fixed_sar is not None and self.fixed_eo is not None
        self.generator.eval()
        with torch.no_grad():
            fake_eo = self.generator(self.fixed_sar)
        self.generator.train()

        def _to_3ch(t: torch.Tensor) -> torch.Tensor:
            return t.expand(-1, 3, -1, -1) if t.shape[1] == 1 else t

        n = min(8, self.fixed_sar.size(0))
        rows = torch.cat(
            [_to_3ch(self.fixed_sar[:n]), _to_3ch(fake_eo[:n]), _to_3ch(self.fixed_eo[:n])]
        )
        save_image(
            make_grid((rows + 1) / 2, nrow=n),
            self.output_dir / "samples" / f"epoch_{epoch:04d}.png",
        )

    # ------------------------------------------------------------------
    # Step orchestration
    # ------------------------------------------------------------------

    def _train_step(self, sar: torch.Tensor, eo: torch.Tensor) -> dict[str, float]:
        """Run one D + G update. Returns merged metrics dict."""
        sar, eo = sar.to(self.device), eo.to(self.device)
        fake_eo = self.generator(sar)
        return {**self._d_step(sar, eo, fake_eo), **self._g_step(sar, eo, fake_eo)}

    @abstractmethod
    def _d_step(
        self, sar: torch.Tensor, eo: torch.Tensor, fake_eo: torch.Tensor
    ) -> dict[str, float]:
        """Discriminator update. Must return a dict containing at least 'd_loss'."""

    @abstractmethod
    def _g_step(
        self, sar: torch.Tensor, eo: torch.Tensor, fake_eo: torch.Tensor
    ) -> dict[str, float]:
        """Generator update. Must return a dict containing at least 'g_adv'."""
