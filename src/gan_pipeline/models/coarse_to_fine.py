"""Coarse-to-fine generator for pix2pixHD (Wang et al., 2018).

Architecture:
  Global stage — 7-level U-Net processes a 2× downsampled input (128×128),
                 producing a coarse prediction at half resolution.
  Local stage  — 3-level encoder-decoder (LocalEnhancer) receives the original
                 full-resolution input concatenated with the upsampled global
                 prediction, and outputs the final refined image.

For 256×256 training data:
  GlobalUNetGenerator: 128×128 input → 128×128 coarse output
  LocalEnhancer:       256×256 × (sar_ch + eo_ch) → 256×256 final output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gan_pipeline.models.base import BaseGenerator


def _enc_block(in_ch: int, out_ch: int, bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _dec_block(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    # ReLU must not be inplace — see CLAUDE.md skip-connection note.
    layers: list[nn.Module] = [
        nn.ReLU(inplace=False),
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


class GlobalUNetGenerator(nn.Module):
    """7-level U-Net for coarse global prediction at half resolution (128×128).

    One level shallower than UNetGenerator, so the bottleneck is 1×1 for
    128×128 inputs rather than requiring 256×256. Dropout in top 3 decoder
    blocks, no BatchNorm in first encoder or bottleneck, Tanh output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 64,
    ) -> None:
        super().__init__()
        bf = base_features

        # Encoder: 7 levels, 128→64→32→16→8→4→2→1
        self.enc1 = _enc_block(in_channels, bf, bn=False)  # 128→64
        self.enc2 = _enc_block(bf, bf * 2)  # 64→32
        self.enc3 = _enc_block(bf * 2, bf * 4)  # 32→16
        self.enc4 = _enc_block(bf * 4, bf * 8)  # 16→8
        self.enc5 = _enc_block(bf * 8, bf * 8)  # 8→4
        self.enc6 = _enc_block(bf * 8, bf * 8)  # 4→2
        self.enc7 = _enc_block(bf * 8, bf * 8, bn=False)  # 2→1 (bottleneck)

        # Decoder: 6 blocks + output conv, 1→2→4→8→16→32→64→128
        self.dec1 = _dec_block(bf * 8, bf * 8, dropout=True)  # 1→2
        self.dec2 = _dec_block(bf * 16, bf * 8, dropout=True)  # cat(e6): 2→4
        self.dec3 = _dec_block(bf * 16, bf * 8, dropout=True)  # cat(e5): 4→8
        self.dec4 = _dec_block(bf * 16, bf * 4)  # cat(e4): 8→16
        self.dec5 = _dec_block(bf * 8, bf * 2)  # cat(e3): 16→32
        self.dec6 = _dec_block(bf * 4, bf)  # cat(e2): 32→64

        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1),  # cat(e1): 64→128
            nn.Tanh(),
        )

        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        d = self.dec1(e7)
        d = self.dec2(torch.cat([d, e6], dim=1))
        d = self.dec3(torch.cat([d, e5], dim=1))
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.dec5(torch.cat([d, e3], dim=1))
        d = self.dec6(torch.cat([d, e2], dim=1))
        return self.out_conv(torch.cat([d, e1], dim=1))  # type: ignore[no-any-return]


class LocalEnhancer(nn.Module):
    """3-level encoder-decoder for full-resolution local refinement.

    Receives the original full-resolution input concatenated with the upsampled
    global prediction and outputs the refined image at the same resolution.
    Minimum spatial size: 8×8 (3 stride-2 encoder blocks).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_features: int = 32,
    ) -> None:
        super().__init__()
        lbf = base_features

        # Encoder: 3 levels
        self.enc1 = _enc_block(in_channels, lbf, bn=False)
        self.enc2 = _enc_block(lbf, lbf * 2)
        self.enc3 = _enc_block(lbf * 2, lbf * 4)

        # Decoder: 2 blocks + output conv with skip connections
        self.dec1 = _dec_block(lbf * 4, lbf * 2)
        self.dec2 = _dec_block(lbf * 4, lbf)  # cat(e2): lbf*2 + lbf*2 = lbf*4

        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(lbf * 2, out_channels, 4, 2, 1),  # cat(e1): lbf + lbf
            nn.Tanh(),
        )

        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d = self.dec1(e3)
        d = self.dec2(torch.cat([d, e2], dim=1))
        return self.out_conv(torch.cat([d, e1], dim=1))  # type: ignore[no-any-return]


class CoarseToFineGenerator(BaseGenerator):
    """Coarse-to-fine conditional image generator (pix2pixHD, Wang et al. 2018).

    Global stage: GlobalUNetGenerator runs on a 2× downsampled input (128×128
    for 256×256 training data), producing a coarse prediction.
    Local stage:  LocalEnhancer receives the full-resolution input concatenated
    with the upsampled coarse prediction and outputs the final image.

    Unlike the original pix2pixHD paper (which uses ResNet-based generators),
    both stages here use U-Net style encoder-decoders, keeping the architecture
    consistent with the pix2pix backbone.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 64,
        local_base_features: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = 0  # conditioned on source image; no latent vector
        self.global_generator = GlobalUNetGenerator(in_channels, out_channels, base_features)
        self.local_enhancer = LocalEnhancer(
            in_channels + out_channels,
            out_channels,
            local_base_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = F.avg_pool2d(x, kernel_size=2, stride=2)
        coarse = self.global_generator(x_low)
        coarse_up = F.interpolate(coarse, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.local_enhancer(torch.cat([x, coarse_up], dim=1))  # type: ignore[no-any-return]
