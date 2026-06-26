"""Paired-image transform factories for pix2pix / pix2pixHD data pipelines."""

import random
from collections.abc import Callable

import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

# Callable that takes (sar_pil, eo_pil) and returns the augmented pair.
# Synchronisation (same crop / flip for both images) is the caller's responsibility.
PairedTransform = Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]


def train_transform(image_size: int) -> PairedTransform:
    """Synchronized resize → random crop → random hflip (pix2pix training augmentation).

    load_size is set to image_size × 1.12 (≈ 286 for 256-px target), matching the
    original pix2pix paper convention.
    """
    load_size = int(image_size * 1.12)

    def _transform(sar: Image.Image, eo: Image.Image) -> tuple[Image.Image, Image.Image]:
        sar = TF.resize(sar, [load_size, load_size], interpolation=TF.InterpolationMode.BICUBIC)
        eo = TF.resize(eo, [load_size, load_size], interpolation=TF.InterpolationMode.BICUBIC)
        i, j, th, tw = transforms.RandomCrop.get_params(sar, (image_size, image_size))
        sar = TF.crop(sar, i, j, th, tw)
        eo = TF.crop(eo, i, j, th, tw)
        if random.random() > 0.5:
            sar = TF.hflip(sar)
            eo = TF.hflip(eo)
        return sar, eo

    return _transform


def val_transform(image_size: int) -> PairedTransform:
    """Resize both images to image_size × image_size (no randomness)."""

    def _transform(sar: Image.Image, eo: Image.Image) -> tuple[Image.Image, Image.Image]:
        sar = TF.resize(sar, [image_size, image_size], interpolation=TF.InterpolationMode.BICUBIC)
        eo = TF.resize(eo, [image_size, image_size], interpolation=TF.InterpolationMode.BICUBIC)
        return sar, eo

    return _transform
