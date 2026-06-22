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
from gan_pipeline.models.resnet_gen import ResnetBlock, ResNetGenerator


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


# --- ResNetGenerator ---


@pytest.mark.parametrize("sar_ch,eo_ch", [(1, 3), (3, 3)])
def test_resnet_generator_output_shape(sar_ch: int, eo_ch: int) -> None:
    g = ResNetGenerator(
        in_channels=sar_ch, out_channels=eo_ch, ngf=16, n_downsampling=2, n_blocks=3
    )
    x = torch.randn(1, sar_ch, 64, 64)
    out = g(x)
    assert out.shape == (1, eo_ch, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_resnet_generator_gradients_flow() -> None:
    g = ResNetGenerator(in_channels=1, out_channels=3, ngf=16, n_downsampling=2, n_blocks=3)
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    out = g(x)
    out.sum().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in g.parameters())


@pytest.mark.parametrize("n_downsampling,n_blocks", [(2, 3), (3, 9)])
def test_resnet_generator_256(n_downsampling: int, n_blocks: int) -> None:
    g = ResNetGenerator(
        in_channels=1, out_channels=3, ngf=16, n_downsampling=n_downsampling, n_blocks=n_blocks
    )
    out = g(torch.randn(1, 1, 256, 256))
    assert out.shape == (1, 3, 256, 256)


def test_resnet_block_residual() -> None:
    block = ResnetBlock(dim=32)
    x = torch.randn(1, 32, 16, 16)
    out = block(x)
    assert out.shape == x.shape
    # output differs from input (block is not zero-initialised)
    assert not torch.allclose(out, x)


def test_vgg_perceptual_loss_missing_weights(tmp_path: Path) -> None:
    import torchvision.models as tvm

    from gan_pipeline.models.losses import VGGPerceptualLoss

    dummy_vgg = tvm.vgg16(weights=None)
    with patch("torchvision.models.vgg16", return_value=dummy_vgg):
        with pytest.raises(FileNotFoundError):
            VGGPerceptualLoss(weights_path=str(tmp_path / "nonexistent.pth"))
