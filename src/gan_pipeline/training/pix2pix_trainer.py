from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.models.losses import (
    LossType,
    VGGPerceptualLoss,
    feature_matching_loss,
    multiscale_discriminator_loss,
    multiscale_generator_loss,
    multiscale_gradient_penalty,
    r1_gradient_penalty,
)
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.training.base_trainer import BaseTrainer


def _make_lr_lambda(n_epochs_keep: int, n_epochs_decay: int) -> Callable[[int], float]:
    """Returns a lambda that keeps LR constant for n_epochs_keep, then linearly decays to 0."""

    def _lambda(epoch: int) -> float:
        return 1.0 - max(0, epoch - n_epochs_keep) / float(n_epochs_decay + 1)

    return _lambda


class Pix2PixTrainer(BaseTrainer):
    """
    Trainer for conditional SAR→EO translation using pix2pix with multi-scale PatchGAN.

    Generator:     G(sar) → fake_eo
    Discriminator: MultiScaleDiscriminator(cat([sar, eo])) → list of patch maps
    Loss:          L_D = mean(hinge/bce across scales)
                   L_G = mean(adv across scales) + lambda_L1 * L1(fake_eo, real_eo)
    """

    # Narrow the discriminator type so pix2pix methods can call forward_with_features.
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
        self.lambda_l1: float = cfg.training.lambda_l1
        self.lambda_vgg: float = float(cfg.training.get("lambda_vgg", 0.0))
        _vgg_weights_path: str | None = cfg.training.get("vgg_weights_path", None)
        self.vgg_loss: VGGPerceptualLoss | None = (
            VGGPerceptualLoss(weights_path=_vgg_weights_path).to(device)
            if self.lambda_vgg > 0
            else None
        )
        self.lambda_fm: float = float(cfg.training.get("lambda_fm", 0.0))
        self.lambda_gp: float = float(cfg.training.get("lambda_gp", 0.0))
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
    # Abstract implementations
    # ------------------------------------------------------------------

    def _log_params(self) -> dict[str, object]:
        return {
            "model": self.cfg.model.name,
            "loss_type": self.cfg.training.loss_type,
            "lambda_l1": self.lambda_l1,
            "lambda_vgg": self.lambda_vgg,
            "lambda_fm": self.lambda_fm,
            "lambda_gp": self.lambda_gp,
            "n_scales": len(self._ms_disc.discriminators),
            "lr_g": self.cfg.training.lr_generator,
            "lr_d": self.cfg.training.lr_discriminator,
            "batch_size": self.cfg.training.batch_size,
        }

    def _step_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sar: torch.Tensor = batch["sar"]
        eo: torch.Tensor = batch["eo"]

        if self.fixed_sar is None:
            self.fixed_sar = sar[:8].to(self.device)
            self.fixed_eo = eo[:8].to(self.device)

        d_loss, g_adv, g_l1, g_vgg, g_fm, d_gp = self._train_step(sar, eo)
        return {
            "d_loss": d_loss,
            "g_adv": g_adv,
            "g_l1": g_l1,
            "g_vgg": g_vgg,
            "g_fm": g_fm,
            "d_gp": d_gp,
        }

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
            [
                _to_3ch(self.fixed_sar[:n]),
                _to_3ch(fake_eo[:n]),
                _to_3ch(self.fixed_eo[:n]),
            ]
        )
        save_image(
            make_grid((rows + 1) / 2, nrow=n),
            self.output_dir / "samples" / f"epoch_{epoch:04d}.png",
        )

    # ------------------------------------------------------------------
    # Core pix2pix step (kept for direct testability)
    # ------------------------------------------------------------------

    def _train_step(
        self, sar: torch.Tensor, eo: torch.Tensor
    ) -> tuple[float, float, float, float, float, float]:
        sar = sar.to(self.device)
        eo = eo.to(self.device)

        fake_eo = self.generator(sar)

        # --- Discriminator (multi-scale) ---
        real_pair = torch.cat([sar, eo], dim=1)
        fake_pair = torch.cat([sar, fake_eo.detach()], dim=1)

        real_maps = self._ms_disc(real_pair)
        fake_maps = self._ms_disc(fake_pair)

        d_loss = multiscale_discriminator_loss(
            real_maps, fake_maps, self.loss_type, self.label_smoothing
        )

        d_gp_val = 0.0
        if self.lambda_gp > 0:
            if self.loss_type == LossType.WASSERSTEIN:
                gp = multiscale_gradient_penalty(self._ms_disc, real_pair, fake_pair)
                d_loss = d_loss + self.lambda_gp * gp
            else:
                gp = r1_gradient_penalty(self._ms_disc, real_pair)
                d_loss = d_loss + (self.lambda_gp / 2) * gp
            d_gp_val = gp.item()

        self.opt_d.zero_grad()
        d_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_d.step()

        # --- Generator ---
        fake_pair_g = torch.cat([sar, fake_eo], dim=1)

        g_fm_val = 0.0
        if self.lambda_fm > 0:
            with torch.no_grad():
                _, real_feats = self._ms_disc.forward_with_features(real_pair)
            fake_maps_g, fake_feats = self._ms_disc.forward_with_features(fake_pair_g)
            g_fm = feature_matching_loss(real_feats, fake_feats)
            g_fm_val = g_fm.item()
        else:
            fake_maps_g = self._ms_disc(fake_pair_g)
            g_fm = None

        g_adv = multiscale_generator_loss(fake_maps_g, self.loss_type)
        g_l1 = F.l1_loss(fake_eo, eo)
        g_loss = g_adv + self.lambda_l1 * g_l1

        if g_fm is not None:
            g_loss = g_loss + self.lambda_fm * g_fm

        g_vgg_val = 0.0
        if self.vgg_loss is not None:
            g_vgg = self.vgg_loss(fake_eo, eo)
            g_loss = g_loss + self.lambda_vgg * g_vgg
            g_vgg_val = g_vgg.item()

        self.opt_g.zero_grad()
        g_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_g.step()

        return d_loss.item(), g_adv.item(), g_l1.item(), g_vgg_val, g_fm_val, d_gp_val
