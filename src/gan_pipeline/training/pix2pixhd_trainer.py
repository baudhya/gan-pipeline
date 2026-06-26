from pathlib import Path

import torch
from omegaconf import DictConfig

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.models.losses import (
    VGGPerceptualLoss,
    feature_matching_loss,
    multiscale_discriminator_loss,
    multiscale_generator_loss,
)
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.training.paired_trainer import PairedGANTrainer


class Pix2PixHDTrainer(PairedGANTrainer):
    """
    pix2pixHD: ResNet generator + 3-scale PatchGAN + VGG + feature matching.

    Loss: LSGAN/hinge + λ_VGG·VGG + λ_FM·FM.
    No L1 and no gradient penalty — spectral norm on D provides stability.
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

        self.lambda_vgg: float = float(cfg.training.get("lambda_vgg", 0.0))
        _vgg_path: str | None = cfg.training.get("vgg_weights_path", None)
        self.vgg_loss: VGGPerceptualLoss | None = (
            VGGPerceptualLoss(weights_path=_vgg_path).to(device) if self.lambda_vgg > 0 else None
        )
        self.lambda_fm: float = float(cfg.training.get("lambda_fm", 0.0))

    def _log_params(self) -> dict[str, object]:
        return {
            "model": self.cfg.model.name,
            "loss_type": self.cfg.training.loss_type,
            "lambda_vgg": self.lambda_vgg,
            "lambda_fm": self.lambda_fm,
            "n_scales": len(self._ms_disc.discriminators),
            "ngf": self.cfg.model.generator.ngf,
            "n_downsampling": self.cfg.model.generator.n_downsampling,
            "n_blocks": self.cfg.model.generator.n_blocks,
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

        self.opt_d.zero_grad()
        d_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_d.step()
        return {"d_loss": d_loss.item()}

    def _g_step(
        self, sar: torch.Tensor, eo: torch.Tensor, fake_eo: torch.Tensor
    ) -> dict[str, float]:
        real_pair = torch.cat([sar, eo], dim=1)
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
        g_loss = g_adv
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
        return {"g_adv": g_adv.item(), "g_vgg": g_vgg_val, "g_fm": g_fm_val}
