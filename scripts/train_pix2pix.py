"""SAR→EO training entry point (pix2pix: U-Net generator + PatchGAN discriminator)."""
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from gan_pipeline.data.paired_dataset import get_paired_dataloader
from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.models.unet import UNetGenerator
from gan_pipeline.training.pix2pix_trainer import Pix2PixTrainer
from gan_pipeline.utils import setup_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    generator = UNetGenerator(
        in_channels=cfg.data.sar_channels,
        out_channels=cfg.data.eo_channels,
        base_features=cfg.model.generator.base_features,
    )
    discriminator = MultiScaleDiscriminator(
        sar_channels=cfg.data.sar_channels,
        eo_channels=cfg.data.eo_channels,
        base_features=cfg.model.discriminator.base_features,
        n_scales=cfg.model.discriminator.n_scales,
    )

    logger.info(f"Generator  params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    trainer = Pix2PixTrainer(generator, discriminator, cfg, device, output_dir)

    if cfg.resume:
        trainer.resume(Path(cfg.resume))

    trainer.train(train_loader)


if __name__ == "__main__":
    main()
