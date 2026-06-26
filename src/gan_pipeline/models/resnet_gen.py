"""ResNet-based generator from pix2pixHD (Wang et al., 2018).

Architecture (n_downsampling=3, n_blocks=9, ngf=64):
    ReflectPad(3) → Conv(7×7, ngf) → Norm → ReLU
    → [Conv(3×3, s2) → Norm → ReLU] × n_downsampling          (encoder)
    → [ResnetBlock(ngf * 2^n_downsampling)] × n_blocks          (ResNet)
    → [ConvTranspose(3×3, s2) → Norm → ReLU] × n_downsampling  (decoder)
    → ReflectPad(3) → Conv(7×7, out_ch) → Tanh

For 256×256 inputs with n_downsampling=3 the bottleneck is 32×32 with ngf*8=512 channels.
"""

import torch
import torch.nn as nn

from gan_pipeline.models.base import BaseGenerator


def _norm(num_channels: int) -> nn.InstanceNorm2d:
    return nn.InstanceNorm2d(num_channels, affine=True)


class ResnetBlock(nn.Module):
    """Residual block with reflection padding (no zero-padding artefacts at borders)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            _norm(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            _norm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # type: ignore[no-any-return]


class ResNetGenerator(BaseGenerator):
    """
    Global ResNet generator from pix2pixHD (Wang et al. 2018).

    Input shape:  (B, in_channels, H, W)
    Output shape: (B, out_channels, H, W) in [-1, 1]

    Default params match the paper's global generator for 256×256 images:
      ngf=64, n_downsampling=3, n_blocks=9
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        ngf: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()
        self.latent_dim = 0

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, bias=False),
            _norm(ngf),
            nn.ReLU(inplace=True),
        ]

        for i in range(n_downsampling):
            c_in = ngf * (2**i)
            c_out = ngf * (2 ** (i + 1))
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False),
                _norm(c_out),
                nn.ReLU(inplace=True),
            ]

        c_bottleneck = ngf * (2**n_downsampling)
        for _ in range(n_blocks):
            layers.append(ResnetBlock(c_bottleneck))

        for i in range(n_downsampling):
            c_in = ngf * (2 ** (n_downsampling - i))
            c_out = ngf * (2 ** (n_downsampling - i - 1))
            layers += [
                nn.ConvTranspose2d(
                    c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
                ),
                _norm(c_out),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]
