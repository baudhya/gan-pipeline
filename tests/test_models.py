from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from gan_pipeline.models import DCGANDiscriminator, DCGANGenerator
from gan_pipeline.models.losses import (
    LossType,
    discriminator_loss,
    generator_loss,
    gradient_penalty,
)


@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_generator_output_shape(image_size: int) -> None:
    g = DCGANGenerator(latent_dim=100, channels=3, image_size=image_size)
    z = torch.randn(4, 100)
    out = g(z)
    assert out.shape == (4, 3, image_size, image_size)
    assert out.min() >= -1.0 and out.max() <= 1.0


@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_discriminator_output_shape(image_size: int) -> None:
    d = DCGANDiscriminator(channels=3, image_size=image_size)
    x = torch.randn(4, 3, image_size, image_size)
    out = d(x)
    assert out.shape == (4,)


@pytest.mark.parametrize("loss_type", list(LossType))
def test_generator_loss(loss_type: LossType) -> None:
    loss = generator_loss(torch.randn(8), loss_type)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss_type", list(LossType))
def test_discriminator_loss(loss_type: LossType) -> None:
    loss = discriminator_loss(torch.randn(8), torch.randn(8), loss_type)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


def test_gradient_penalty() -> None:
    d = DCGANDiscriminator(channels=3, image_size=64)
    gp = gradient_penalty(
        d, torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64), torch.device("cpu")
    )
    assert gp.shape == torch.Size([])
    assert torch.isfinite(gp)


def test_generator_sample() -> None:
    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    samples = g.sample(8, torch.device("cpu"))
    assert samples.shape == (8, 3, 64, 64)


def test_generator_loss_invalid_type() -> None:
    with pytest.raises(ValueError):
        generator_loss(torch.randn(8), MagicMock())  # type: ignore[arg-type]


def test_discriminator_loss_invalid_type() -> None:
    with pytest.raises(ValueError):
        discriminator_loss(torch.randn(8), torch.randn(8), MagicMock())  # type: ignore[arg-type]


def test_vgg_perceptual_loss_missing_weights(tmp_path: Path) -> None:
    import torchvision.models as tvm

    from gan_pipeline.models.losses import VGGPerceptualLoss

    dummy_vgg = tvm.vgg16(weights=None)
    with patch("torchvision.models.vgg16", return_value=dummy_vgg):
        with pytest.raises(FileNotFoundError):
            VGGPerceptualLoss(weights_path=str(tmp_path / "nonexistent.pth"))
