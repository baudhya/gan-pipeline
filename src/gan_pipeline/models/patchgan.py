"""70×70 PatchGAN discriminator for pix2pix-style conditional image translation."""

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from gan_pipeline.models.base import BaseDiscriminator


def _patch_block(in_ch: int, out_ch: int, stride: int = 2, bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not bn)]
    if bn:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGANDiscriminator(BaseDiscriminator):
    """
    Classifies overlapping 70×70 patches as real or fake.
    Input:  concatenation of SAR and EO images — (sar_ch + eo_ch, H, W)
    Output: patch logit map — (1, H', W')  where H'≈W'≈30 for 256×256 input.

    Each output logit represents one 70×70 receptive-field patch.
    BCE is applied element-wise and averaged across all patches.
    """

    def __init__(
        self,
        sar_channels: int = 1,
        eo_channels: int = 3,
        base_features: int = 64,
        spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        bf = base_features
        in_ch = sar_channels + eo_channels

        self.net = nn.Sequential(
            _patch_block(in_ch, bf, stride=2, bn=False),  # 256→128
            _patch_block(bf, bf * 2, stride=2),  # 128→64
            _patch_block(bf * 2, bf * 4, stride=2),  # 64→32
            _patch_block(bf * 4, bf * 8, stride=1),  # 32→31 (stride=1, no downsampling)
            nn.Conv2d(bf * 8, 1, 4, 1, 1),  # 31→30 (raw logits)
        )

        self._init_weights()
        if spectral_norm:
            self._apply_spectral_norm()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def _apply_spectral_norm(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn_utils.spectral_norm(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: pre-concatenated (sar, eo) tensor of shape (B, sar_ch+eo_ch, H, W)."""
        return self.net(x)  # type: ignore[no-any-return]

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the network and return (logit_map, [intermediate_feature, ...]).

        Features are the outputs after each conv block (all layers except the
        final 1-channel conv). Used by feature_matching_loss.
        """
        features: list[torch.Tensor] = []
        layers = list(self.net.children())
        for layer in layers[:-1]:
            x = layer(x)
            features.append(x)
        logit = layers[-1](x)
        return logit, features
