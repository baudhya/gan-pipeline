from pathlib import Path

import torch
from loguru import logger

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
