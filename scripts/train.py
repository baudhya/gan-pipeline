"""Main training entry point."""
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from gan_pipeline.data import get_dataloader
from gan_pipeline.models import DCGANDiscriminator, DCGANGenerator
from gan_pipeline.training import GANTrainer
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

    dataloader = get_dataloader(
        root=cfg.data.root,
        image_size=cfg.data.image_size,
        mean=list(cfg.data.mean),
        std=list(cfg.data.std),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
    )
    logger.info(f"Dataset: {len(dataloader.dataset)} images")

    generator = DCGANGenerator(
        latent_dim=cfg.model.latent_dim,
        channels=cfg.data.channels,
        base_features=cfg.model.generator.base_features,
        image_size=cfg.data.image_size,
    )
    discriminator = DCGANDiscriminator(
        channels=cfg.data.channels,
        base_features=cfg.model.discriminator.base_features,
        image_size=cfg.data.image_size,
    )

    logger.info(f"Generator  params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    trainer = GANTrainer(generator, discriminator, cfg, device, output_dir)

    if cfg.resume:
        trainer.resume(Path(cfg.resume))

    trainer.train(dataloader)


if __name__ == "__main__":
    main()
