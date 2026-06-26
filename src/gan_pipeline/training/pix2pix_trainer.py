from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.models.losses import (
    LossType,
    multiscale_discriminator_loss,
    multiscale_generator_loss,
    multiscale_gradient_penalty,
    r1_gradient_penalty,
)
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.training.paired_trainer import PairedGANTrainer


class Pix2PixTrainer(PairedGANTrainer):
    """
    Original pix2pix: U-Net generator + single-scale PatchGAN.

    Loss: BCE + λ_L1·L1 + optional R1 gradient penalty.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: MultiScaleDiscriminator,
        cfg: DictConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        super().__init__(generator, discriminator, cfg, device, output_dir)
        self.lambda_l1: float = cfg.training.lambda_l1
        self.lambda_gp: float = float(cfg.training.get("lambda_gp", 0.0))

    def _log_params(self) -> dict[str, object]:
        return {
            "model": self.cfg.model.name,
            "loss_type": self.cfg.training.loss_type,
            "lambda_l1": self.lambda_l1,
            "lambda_gp": self.lambda_gp,
            "n_scales": len(self._ms_disc.discriminators),
            "base_features": self.cfg.model.generator.base_features,
            "lr_g": self.cfg.training.lr_generator,
            "lr_d": self.cfg.training.lr_discriminator,
            "batch_size": self.cfg.training.batch_size,
        }

    def _d_step(
        self, sar: torch.Tensor, eo: torch.Tensor, fake_eo: torch.Tensor
    ) -> dict[str, float]:
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
        return {"d_loss": d_loss.item(), "d_gp": d_gp_val}

    def _g_step(
        self, sar: torch.Tensor, eo: torch.Tensor, fake_eo: torch.Tensor
    ) -> dict[str, float]:
        fake_pair_g = torch.cat([sar, fake_eo], dim=1)
        fake_maps_g = self._ms_disc(fake_pair_g)

        g_adv = multiscale_generator_loss(fake_maps_g, self.loss_type)
        g_l1 = F.l1_loss(fake_eo, eo)
        g_loss = g_adv + self.lambda_l1 * g_l1

        self.opt_g.zero_grad()
        g_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_g.step()
        return {"g_adv": g_adv.item(), "g_l1": g_l1.item()}
