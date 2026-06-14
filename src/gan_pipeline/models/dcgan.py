import math

import torch
import torch.nn as nn

from gan_pipeline.models.base import BaseDiscriminator, BaseGenerator


def _conv_block(in_ch: int, out_ch: int, bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _deconv_block(in_ch: int, out_ch: int, last: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
    if not last:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    else:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class DCGANGenerator(BaseGenerator):
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 3,
        base_features: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        n_up = int(math.log2(image_size)) - 2
        init_f = base_features * (2 ** (n_up - 1))

        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, init_f, 4, 1, 0, bias=False),
            nn.BatchNorm2d(init_f),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        in_f = init_f
        for _ in range(n_up - 1):
            out_f = in_f // 2
            blocks.append(_deconv_block(in_f, out_f))
            in_f = out_f
        blocks.append(_deconv_block(in_f, channels, last=True))
        self.blocks = nn.Sequential(*blocks)

        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.project(z.view(z.size(0), -1, 1, 1)))  # type: ignore[no-any-return]


class DCGANDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        channels: int = 3,
        base_features: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()

        n_down = int(math.log2(image_size)) - 2

        layers: list[nn.Module] = [_conv_block(channels, base_features, bn=False)]
        in_f = base_features
        for _ in range(n_down - 1):
            out_f = in_f * 2
            layers.append(_conv_block(in_f, out_f))
            in_f = out_f
        layers.append(nn.Conv2d(in_f, 1, 4, 1, 0, bias=False))
        self.net = nn.Sequential(*layers)

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # type: ignore[no-any-return]
