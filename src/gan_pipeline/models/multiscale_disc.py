"""Multi-scale PatchGAN discriminator (pix2pix HD style)."""

import torch
import torch.nn as nn

from gan_pipeline.models.patchgan import PatchGANDiscriminator


class MultiScaleDiscriminator(nn.Module):
    """
    Runs N independent PatchGAN discriminators on N progressively coarser scales.
    Each scale is 2× downsampled from the previous (AvgPool).

    Finest scale sees full-resolution detail; coarser scales see global structure.
    All discriminators share the same architecture but have separate weights.

    n_scales=1 is equivalent to a single-scale PatchGAN.
    """

    def __init__(
        self,
        sar_channels: int = 1,
        eo_channels: int = 3,
        base_features: int = 64,
        n_scales: int = 3,
    ) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(sar_channels, eo_channels, base_features)
            for _ in range(n_scales)
        ])
        # Smooth downsampling between scales; count_include_pad=False avoids border artifacts
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: cat([SAR, EO]) tensor — (B, sar_ch + eo_ch, H, W)
        Returns:
            List of patch logit maps ordered finest → coarsest.
            Each element has shape (B, 1, H_i, W_i).
        """
        outputs: list[torch.Tensor] = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs
