from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from gan_pipeline.data.transforms import get_transforms


def get_dataloader(
    root: str,
    image_size: int,
    mean: list[float],
    std: list[float],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:  # type: ignore[type-arg]
    transform = get_transforms(image_size, mean, std)
    dataset = ImageFolder(root=str(Path(root)), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
