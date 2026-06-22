"""U-Net generator for pix2pix (256x256, 8-level encoder/decoder with skip connections)."""

import torch
import torch.nn as nn

from gan_pipeline.models.base import BaseGenerator


def _enc_block(in_ch: int, out_ch: int, bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
    if bn:
        layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _dec_block(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    # ReLU must not be inplace: the encoder skip tensors are reused and
    # inplace modification would corrupt saved activations for backward.
    layers: list[nn.Module] = [
        nn.ReLU(inplace=False),
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(BaseGenerator):
    """
    U-Net generator conditioned on a source image (e.g. SAR → EO).
    Input:  (sar_channels, 256, 256)
    Output: (eo_channels,  256, 256) in range [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_features: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = 0  # not used; generator is conditioned on source image
        bf = base_features

        # Encoder (each block halves spatial dims)
        self.enc1 = _enc_block(in_channels, bf, bn=False)  # 256→128, 64
        self.enc2 = _enc_block(bf, bf * 2)  # 128→64,  128
        self.enc3 = _enc_block(bf * 2, bf * 4)  # 64→32,   256
        self.enc4 = _enc_block(bf * 4, bf * 8)  # 32→16,   512
        self.enc5 = _enc_block(bf * 8, bf * 8)  # 16→8,    512
        self.enc6 = _enc_block(bf * 8, bf * 8)  # 8→4,     512
        self.enc7 = _enc_block(bf * 8, bf * 8)  # 4→2,     512
        self.enc8 = _enc_block(bf * 8, bf * 8, bn=False)  # 2→1,     512 (bottleneck)

        # Decoder with skip connections (concat doubles the in_channels after first block)
        self.dec1 = _dec_block(bf * 8, bf * 8, dropout=True)  # 1→2
        self.dec2 = _dec_block(bf * 16, bf * 8, dropout=True)  # 2→4
        self.dec3 = _dec_block(bf * 16, bf * 8, dropout=True)  # 4→8
        self.dec4 = _dec_block(bf * 16, bf * 8)  # 8→16
        self.dec5 = _dec_block(bf * 16, bf * 4)  # 16→32
        self.dec6 = _dec_block(bf * 8, bf * 2)  # 32→64
        self.dec7 = _dec_block(bf * 4, bf)  # 64→128

        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1),  # 128→256
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.GroupNorm) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # (bf,    128, 128)
        e2 = self.enc2(e1)  # (bf*2,  64,  64)
        e3 = self.enc3(e2)  # (bf*4,  32,  32)
        e4 = self.enc4(e3)  # (bf*8,  16,  16)
        e5 = self.enc5(e4)  # (bf*8,  8,   8)
        e6 = self.enc6(e5)  # (bf*8,  4,   4)
        e7 = self.enc7(e6)  # (bf*8,  2,   2)
        e8 = self.enc8(e7)  # (bf*8,  1,   1)

        d = self.dec1(e8)  # (bf*8,  2,   2)
        d = self.dec2(torch.cat([d, e7], dim=1))  # (bf*8,  4,   4)
        d = self.dec3(torch.cat([d, e6], dim=1))  # (bf*8,  8,   8)
        d = self.dec4(torch.cat([d, e5], dim=1))  # (bf*8,  16,  16)
        d = self.dec5(torch.cat([d, e4], dim=1))  # (bf*4,  32,  32)
        d = self.dec6(torch.cat([d, e3], dim=1))  # (bf*2,  64,  64)
        d = self.dec7(torch.cat([d, e2], dim=1))  # (bf,    128, 128)
        return self.out_conv(torch.cat([d, e1], dim=1))  # type: ignore[no-any-return]
