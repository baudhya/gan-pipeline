"""SAR→EO training entry point — pix2pixHD (coarse-to-fine generator, multi-scale PatchGAN)."""

import random
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from gan_pipeline.data.paired_dataset import get_paired_dataloader
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.models.resnet_gen import ResNetGenerator
from gan_pipeline.training.pix2pixhd_trainer import Pix2PixHDTrainer
from gan_pipeline.utils import setup_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config_pix2pixhd")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    set_seed(cfg.seed)

    _device_cfg: str = cfg.get("device", "auto")
    if _device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(_device_cfg)
    logger.info(f"Device: {device}")

    train_loader = get_paired_dataloader(
        root=cfg.data.root,
        split="train",
        image_size=cfg.data.image_size,
        sar_channels=cfg.data.sar_channels,
        eo_channels=cfg.data.eo_channels,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        augment=cfg.data.augment_train,
        dataset_format=cfg.data.dataset_format,
    )
    logger.info(f"Train set: {len(train_loader.dataset)} pairs")

    generator = ResNetGenerator(
        in_channels=cfg.data.sar_channels,
        out_channels=cfg.data.eo_channels,
        ngf=cfg.model.generator.ngf,
        n_downsampling=cfg.model.generator.n_downsampling,
        n_blocks=cfg.model.generator.n_blocks,
    )
    discriminator = MultiScaleDiscriminator(
        sar_channels=cfg.data.sar_channels,
        eo_channels=cfg.data.eo_channels,
        base_features=cfg.model.discriminator.base_features,
        n_scales=cfg.model.discriminator.n_scales,
        spectral_norm=cfg.model.discriminator.get("spectral_norm", False),
    )

    logger.info(f"Generator  params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    trainer = Pix2PixHDTrainer(generator, discriminator, cfg, device, output_dir)

    if cfg.resume:
        trainer.resume(Path(cfg.resume))

    trainer.train(train_loader)


if __name__ == "__main__":
    main()
