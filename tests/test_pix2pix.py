"""Tests for pix2pix: U-Net, PatchGAN, multi-scale discriminator, paired dataset, trainer."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator
from gan_pipeline.models.patchgan import PatchGANDiscriminator
from gan_pipeline.models.unet import UNetGenerator

# --- U-Net Generator ---

@pytest.mark.parametrize("sar_ch,eo_ch", [(1, 3), (3, 3), (1, 1)])
def test_unet_output_shape(sar_ch: int, eo_ch: int) -> None:
    g = UNetGenerator(in_channels=sar_ch, out_channels=eo_ch)
    x = torch.randn(2, sar_ch, 256, 256)
    out = g(x)
    assert out.shape == (2, eo_ch, 256, 256)
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh


def test_unet_skip_connections_preserve_gradients() -> None:
    g = UNetGenerator(in_channels=1, out_channels=3)
    x = torch.randn(1, 1, 256, 256, requires_grad=True)
    out = g(x)
    out.sum().backward()
    assert x.grad is not None


# --- PatchGAN Discriminator ---

@pytest.mark.parametrize("sar_ch,eo_ch", [(1, 3), (3, 3)])
def test_patchgan_output_shape(sar_ch: int, eo_ch: int) -> None:
    d = PatchGANDiscriminator(sar_channels=sar_ch, eo_channels=eo_ch)
    x = torch.randn(2, sar_ch + eo_ch, 256, 256)
    out = d(x)
    # Output should be a 2D patch map (B, 1, H', W')
    assert out.ndim == 4
    assert out.shape[0] == 2 and out.shape[1] == 1


def test_patchgan_patch_size_256() -> None:
    """For 256×256 input the 70×70 PatchGAN should output ~30×30."""
    d = PatchGANDiscriminator(sar_channels=1, eo_channels=3)
    x = torch.randn(1, 4, 256, 256)
    out = d(x)
    # Allow ±2 tolerance around 30
    assert 28 <= out.shape[-1] <= 32
    assert 28 <= out.shape[-2] <= 32


# --- MultiScaleDiscriminator ---

@pytest.mark.parametrize("n_scales", [1, 2, 3])
def test_multiscale_output_length(n_scales: int) -> None:
    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=n_scales)
    x = torch.randn(2, 4, 256, 256)  # 1+3 channels
    out = d(x)
    assert len(out) == n_scales
    # Each scale should have a smaller spatial size than the previous
    for i in range(1, len(out)):
        assert out[i].shape[-1] < out[i - 1].shape[-1]


def test_multiscale_patch_shapes_256() -> None:
    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=3)
    x = torch.randn(1, 4, 256, 256)
    maps = d(x)
    # Scale 0: full 256 input  → ~30×30 patches
    # Scale 1: 128 input       → ~14×14 patches
    # Scale 2: 64 input        → ~6×6  patches
    assert maps[0].shape[-1] > maps[1].shape[-1] > maps[2].shape[-1]


def test_vgg_perceptual_loss_offline_path(tmp_path: Path) -> None:
    """VGGPerceptualLoss loads weights from a local file (no network needed)."""
    import torchvision.models as tvm

    from gan_pipeline.models.losses import VGGPerceptualLoss

    # Save a random-weight VGG16 state dict to disk
    weights_file = tmp_path / "vgg16.pth"
    torch.save(tvm.vgg16(weights=None).state_dict(), weights_file)

    loss_fn = VGGPerceptualLoss(weights_path=str(weights_file))
    fake = torch.randn(1, 3, 64, 64)
    real = torch.randn(1, 3, 64, 64)
    loss = loss_fn(fake, real)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_vgg_perceptual_loss(channels: int) -> None:
    """VGGPerceptualLoss: correct scalar output, finite, zero on identical inputs."""
    from unittest.mock import patch

    import torchvision.models as tvm

    dummy_vgg = tvm.vgg16(weights=None)  # random weights — no network download
    with patch("torchvision.models.vgg16", return_value=dummy_vgg):
        from gan_pipeline.models.losses import VGGPerceptualLoss
        loss_fn = VGGPerceptualLoss()

    fake = torch.randn(2, channels, 64, 64)
    real = torch.randn(2, channels, 64, 64)

    loss = loss_fn(fake, real)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Identical inputs → zero loss
    zero_loss = loss_fn(fake, fake)
    assert zero_loss.item() < 1e-5


@pytest.mark.parametrize("spectral_norm", [True, False])
def test_patchgan_spectral_norm(spectral_norm: bool) -> None:
    d = PatchGANDiscriminator(sar_channels=1, eo_channels=3, spectral_norm=spectral_norm)
    conv_layers = [m for m in d.modules() if isinstance(m, nn.Conv2d)]

    if spectral_norm:
        assert all(hasattr(m, "weight_orig") for m in conv_layers), \
            "All Conv2d layers should have weight_orig after spectral_norm"
    else:
        assert not any(hasattr(m, "weight_orig") for m in conv_layers), \
            "No Conv2d layers should have weight_orig without spectral_norm"

    # Forward pass should work regardless
    x = torch.randn(2, 4, 256, 256)
    out = d(x)
    assert out.shape[1] == 1
    assert torch.isfinite(out).all()


def test_multiscale_spectral_norm_threads_through() -> None:
    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=2, spectral_norm=True)
    for disc in d.discriminators:
        conv_layers = [m for m in disc.modules() if isinstance(m, nn.Conv2d)]
        assert all(hasattr(m, "weight_orig") for m in conv_layers)


def test_patchgan_forward_with_features() -> None:
    d = PatchGANDiscriminator(sar_channels=1, eo_channels=3)
    x = torch.randn(2, 4, 256, 256)
    logit, features = d.forward_with_features(x)

    # Logit map matches plain forward()
    assert logit.shape == d(x).shape
    # 4 conv blocks → 4 feature tensors, each 4-D
    assert len(features) == 4
    assert all(f.ndim == 4 for f in features)
    # Features are progressively smaller (each block strides / pools)
    for i in range(1, len(features)):
        assert features[i].shape[-1] <= features[i - 1].shape[-1]


@pytest.mark.parametrize("n_scales", [1, 2, 3])
def test_multiscale_forward_with_features(n_scales: int) -> None:
    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=n_scales)
    x = torch.randn(2, 4, 256, 256)
    logits, all_features = d.forward_with_features(x)

    assert len(logits) == n_scales
    assert len(all_features) == n_scales
    # Each scale has 4 feature tensors (one per conv block)
    for scale_feats in all_features:
        assert len(scale_feats) == 4


def test_feature_matching_loss() -> None:
    from gan_pipeline.models.losses import feature_matching_loss

    # 3 scales × 4 layers; feature shapes don't need to match disc exactly
    real_feats = [[torch.randn(2, 64, 16, 16) for _ in range(4)] for _ in range(3)]
    fake_feats = [[torch.randn(2, 64, 16, 16) for _ in range(4)] for _ in range(3)]

    loss = feature_matching_loss(real_feats, fake_feats)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    # Identical inputs → zero loss (real_feats used for both)
    zero_loss = feature_matching_loss(real_feats, real_feats)
    assert zero_loss.item() < 1e-6


def test_multiscale_discriminator_loss() -> None:
    from gan_pipeline.models.losses import (
        LossType,
        multiscale_discriminator_loss,
        multiscale_generator_loss,
    )

    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=3)
    real = torch.randn(2, 4, 256, 256)
    fake = torch.randn(2, 4, 256, 256)
    real_maps = d(real)
    fake_maps = d(fake)

    for loss_type in LossType:
        d_loss = multiscale_discriminator_loss(real_maps, fake_maps, loss_type)
        g_loss = multiscale_generator_loss(fake_maps, loss_type)
        assert d_loss.shape == torch.Size([])
        assert g_loss.shape == torch.Size([])
        assert torch.isfinite(d_loss) and torch.isfinite(g_loss)


# --- End-to-end pix2pix train step (multi-scale + hinge) ---

@pytest.mark.parametrize("loss_type,n_scales", [("hinge", 3), ("bce", 1), ("hinge", 2)])
def test_pix2pix_train_step(
    loss_type: str, n_scales: int, cfg, device: torch.device, tmp_path: Path
) -> None:
    import omegaconf

    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)
        cfg.training.loss_type = loss_type
        cfg.training.lambda_l1 = 100.0
        cfg.training.lambda_vgg = 0.0  # skip VGG to avoid network download in CI
        cfg.training.lambda_fm = 0.0
        cfg.data.sar_channels = 1
        cfg.data.eo_channels = 3

    from gan_pipeline.training.pix2pix_trainer import Pix2PixTrainer

    g = UNetGenerator(in_channels=1, out_channels=3)
    d = MultiScaleDiscriminator(sar_channels=1, eo_channels=3, n_scales=n_scales)
    trainer = Pix2PixTrainer(g, d, cfg, device, tmp_path)

    sar = torch.randn(2, 1, 256, 256)
    eo = torch.randn(2, 3, 256, 256)
    d_loss, g_adv, g_l1, g_vgg, g_fm = trainer._train_step(sar, eo)

    assert all(isinstance(v, float) for v in [d_loss, g_adv, g_l1, g_vgg, g_fm])
    assert all(not (v != v) for v in [d_loss, g_adv, g_l1, g_vgg, g_fm])  # no NaN
    assert g_vgg == 0.0  # VGG disabled
    assert g_fm == 0.0   # FM disabled


# --- Paired dataset ---

def _make_side_by_side_dir(tmp_path: Path, n: int = 4, sar_mode: str = "L") -> Path:
    """Create a minimal side-by-side paired dataset for testing."""
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    for i in range(n):
        sar = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        if sar_mode == "RGB":
            sar = sar.convert("RGB")
        eo = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        # Combine side-by-side
        w_sar, h = sar.size
        w_eo, _ = eo.size
        combined = Image.new("RGB", (w_sar + w_eo, h))
        combined.paste(sar.convert("RGB"), (0, 0))
        combined.paste(eo, (w_sar, 0))
        combined.save(split_dir / f"{i:04d}.png")
    return tmp_path


def test_side_by_side_dataset(tmp_path: Path) -> None:
    from gan_pipeline.data.paired_dataset import SideBySidePairedDataset

    _make_side_by_side_dir(tmp_path, n=4)
    ds = SideBySidePairedDataset(
        str(tmp_path), "train", image_size=64, sar_channels=1, eo_channels=3, augment=False
    )
    assert len(ds) == 4

    sample = ds[0]
    assert sample["sar"].shape == (1, 64, 64)
    assert sample["eo"].shape == (3, 64, 64)
    assert sample["sar"].min() >= -1.0 and sample["sar"].max() <= 1.0


def test_side_by_side_dataset_augment(tmp_path: Path) -> None:
    from gan_pipeline.data.paired_dataset import SideBySidePairedDataset

    _make_side_by_side_dir(tmp_path, n=2)
    ds = SideBySidePairedDataset(
        str(tmp_path), "train", image_size=64, sar_channels=1, eo_channels=3, augment=True
    )
    sample = ds[0]
    assert sample["sar"].shape == (1, 64, 64)
    assert sample["eo"].shape == (3, 64, 64)
