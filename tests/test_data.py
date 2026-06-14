import numpy as np
import pytest
from PIL import Image

from gan_pipeline.data.transforms import get_transforms


def test_transforms_shape_and_range() -> None:
    transform = get_transforms(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    tensor = transform(img)
    assert tensor.shape == (3, 64, 64)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


@pytest.mark.parametrize("size", [32, 64, 128])
def test_transforms_output_size(size: int) -> None:
    transform = get_transforms(size, [0.5] * 3, [0.5] * 3)
    img = Image.fromarray(np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8))
    assert transform(img).shape == (3, size, size)
