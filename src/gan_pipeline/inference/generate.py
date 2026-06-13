from pathlib import Path

import torch
from loguru import logger
from torchvision.utils import save_image

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.utils.checkpointing import load_checkpoint


def load_generator(
    checkpoint_path: Path,
    generator: BaseGenerator,
    device: torch.device,
) -> BaseGenerator:
    state = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(state["generator"])
    generator.to(device).eval()
    logger.info(f"Loaded generator from {checkpoint_path} (epoch {state['epoch']})")
    return generator


def generate_images(
    generator: BaseGenerator,
    n: int,
    device: torch.device,
    output_dir: Path,
    batch_size: int = 64,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images: list[torch.Tensor] = []
    remaining = n

    with torch.no_grad():
        while remaining > 0:
            bs = min(batch_size, remaining)
            z = torch.randn(bs, generator.latent_dim, device=device)
            all_images.append(generator(z).cpu())
            remaining -= bs

    images = (torch.cat(all_images) + 1) / 2  # [-1,1] -> [0,1]
    out = output_dir / "generated.png"
    save_image(images, out, nrow=int(n**0.5))
    logger.info(f"Saved {n} images to {out}")
    return out
