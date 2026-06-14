"""Generate a grid of images from a trained checkpoint."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from gan_pipeline.inference import generate_images, load_generator
from gan_pipeline.models import DCGANGenerator


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DCGANGenerator(
        latent_dim=cfg.model.latent_dim,
        channels=cfg.data.channels,
        base_features=cfg.model.generator.base_features,
        image_size=cfg.data.image_size,
    )
    generator = load_generator(Path(cfg.checkpoint), generator, device)

    generate_images(
        generator,
        n=cfg.get("n_samples", 64),
        device=device,
        output_dir=Path(cfg.get("gen_output_dir", "outputs/generated")),
    )


if __name__ == "__main__":
    main()
