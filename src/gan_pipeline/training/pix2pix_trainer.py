from collections.abc import Callable
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.models.losses import (
    LossType,
    VGGPerceptualLoss,
    feature_matching_loss,
    multiscale_discriminator_loss,
    multiscale_generator_loss,
    multiscale_gradient_penalty,
)
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.utils.checkpointing import load_checkpoint, save_checkpoint


def _make_lr_lambda(n_epochs_keep: int, n_epochs_decay: int) -> Callable[[int], float]:
    """Returns a lambda that keeps LR constant for n_epochs_keep, then linearly decays to 0."""

    def _lambda(epoch: int) -> float:
        return 1.0 - max(0, epoch - n_epochs_keep) / float(n_epochs_decay + 1)

    return _lambda


class Pix2PixTrainer:
    """
    Trainer for conditional SAR→EO translation using pix2pix with multi-scale PatchGAN.

    Generator:     G(sar) → fake_eo
    Discriminator: MultiScaleDiscriminator(cat([sar, eo])) → list of patch maps
    Loss:          L_D = mean(hinge/bce across scales)
                   L_G = mean(adv across scales) + lambda_L1 * L1(fake_eo, real_eo)
    """

    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: MultiScaleDiscriminator,
        cfg: DictConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
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

        self.opt_g = torch.optim.Adam(
            generator.parameters(),
            lr=cfg.training.lr_generator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=cfg.training.lr_discriminator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )

        n_decay = cfg.training.epochs // 2
        n_keep = cfg.training.epochs - n_decay
        self._lr_lambda = _make_lr_lambda(n_keep, n_decay)
        self.sched_g = LambdaLR(self.opt_g, self._lr_lambda)
        self.sched_d = LambdaLR(self.opt_d, self._lr_lambda)

        self.fixed_sar: torch.Tensor | None = None
        self.fixed_eo: torch.Tensor | None = None
        self.start_epoch = 0

        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def resume(self, checkpoint_path: Path) -> None:
        state = load_checkpoint(checkpoint_path, self.device)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.start_epoch = state["epoch"] + 1
        # Recreate schedulers at the correct epoch so LR is right on restart.
        self.sched_g = LambdaLR(self.opt_g, self._lr_lambda, last_epoch=self.start_epoch - 1)
        self.sched_d = LambdaLR(self.opt_d, self._lr_lambda, last_epoch=self.start_epoch - 1)
        logger.info(f"Resumed from epoch {state['epoch']}")

    def _train_step(
        self, sar: torch.Tensor, eo: torch.Tensor
    ) -> tuple[float, float, float, float, float, float]:
        sar = sar.to(self.device)
        eo = eo.to(self.device)

        fake_eo = self.generator(sar)

        # --- Discriminator (multi-scale) ---
        real_pair = torch.cat([sar, eo], dim=1)
        fake_pair = torch.cat([sar, fake_eo.detach()], dim=1)

        real_maps = self.discriminator(real_pair)  # list[Tensor]
        fake_maps = self.discriminator(fake_pair)

        d_loss = multiscale_discriminator_loss(real_maps, fake_maps, self.loss_type)

        d_gp_val = 0.0
        if self.loss_type == LossType.WASSERSTEIN and self.lambda_gp > 0:
            gp = multiscale_gradient_penalty(self.discriminator, real_pair, fake_pair)
            d_loss = d_loss + self.lambda_gp * gp
            d_gp_val = gp.item()

        self.opt_d.zero_grad()
        d_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_d.step()

        # --- Generator ---
        fake_pair_g = torch.cat([sar, fake_eo], dim=1)

        g_fm_val = 0.0
        if self.lambda_fm > 0:
            with torch.no_grad():
                _, real_feats = self.discriminator.forward_with_features(real_pair)
            fake_maps_g, fake_feats = self.discriminator.forward_with_features(fake_pair_g)
            g_fm = feature_matching_loss(real_feats, fake_feats)
            g_fm_val = g_fm.item()
        else:
            fake_maps_g = self.discriminator(fake_pair_g)
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

    def train(self, dataloader: DataLoader) -> None:  # type: ignore[type-arg]
        n_scales = len(self.discriminator.discriminators)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.cfg.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": self.cfg.model.name,
                    "loss_type": self.cfg.training.loss_type,
                    "lambda_l1": self.lambda_l1,
                    "lambda_vgg": self.lambda_vgg,
                    "lambda_fm": self.lambda_fm,
                    "lambda_gp": self.lambda_gp,
                    "n_scales": n_scales,
                    "lr_g": self.cfg.training.lr_generator,
                    "lr_d": self.cfg.training.lr_discriminator,
                    "batch_size": self.cfg.training.batch_size,
                }
            )

            for epoch in range(self.start_epoch, self.cfg.training.epochs):
                self.generator.train()
                self.discriminator.train()

                d_losses: list[float] = []
                g_adv_losses: list[float] = []
                g_l1_losses: list[float] = []
                g_vgg_losses: list[float] = []
                g_fm_losses: list[float] = []
                d_gp_losses: list[float] = []

                for i, batch in enumerate(dataloader):
                    sar: torch.Tensor = batch["sar"]
                    eo: torch.Tensor = batch["eo"]

                    if self.fixed_sar is None:
                        self.fixed_sar = sar[:8].to(self.device)
                        self.fixed_eo = eo[:8].to(self.device)

                    d_loss, g_adv, g_l1, g_vgg, g_fm, d_gp = self._train_step(sar, eo)
                    d_losses.append(d_loss)
                    g_adv_losses.append(g_adv)
                    g_l1_losses.append(g_l1)
                    g_vgg_losses.append(g_vgg)
                    g_fm_losses.append(g_fm)
                    d_gp_losses.append(d_gp)

                    if i % self.cfg.training.log_every == 0:
                        logger.info(
                            f"Epoch {epoch}/{self.cfg.training.epochs} "
                            f"[{i}/{len(dataloader)}] "
                            f"D: {d_loss:.4f}  G_adv: {g_adv:.4f}  "
                            f"G_L1: {g_l1:.4f}  G_VGG: {g_vgg:.4f}  G_FM: {g_fm:.4f}  "
                            f"D_GP: {d_gp:.4f}"
                        )

                avg_d = sum(d_losses) / len(d_losses)
                avg_g_adv = sum(g_adv_losses) / len(g_adv_losses)
                avg_g_l1 = sum(g_l1_losses) / len(g_l1_losses)
                avg_g_vgg = sum(g_vgg_losses) / len(g_vgg_losses)
                avg_g_fm = sum(g_fm_losses) / len(g_fm_losses)
                avg_d_gp = sum(d_gp_losses) / len(d_gp_losses)

                mlflow.log_metrics(
                    {
                        "d_loss": avg_d,
                        "g_adv": avg_g_adv,
                        "g_l1": avg_g_l1,
                        "g_vgg": avg_g_vgg,
                        "g_fm": avg_g_fm,
                        "d_gp": avg_d_gp,
                    },
                    step=epoch,
                )

                if epoch % self.cfg.training.sample_every == 0:
                    self._save_samples(epoch)

                if epoch % self.cfg.training.save_every == 0:
                    save_checkpoint(
                        self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                        epoch,
                        self.generator,
                        self.discriminator,
                        self.opt_g,
                        self.opt_d,
                        {
                            "d_loss": avg_d,
                            "g_adv": avg_g_adv,
                            "g_l1": avg_g_l1,
                            "g_vgg": avg_g_vgg,
                            "g_fm": avg_g_fm,
                        },
                    )

                self.sched_g.step()
                self.sched_d.step()
