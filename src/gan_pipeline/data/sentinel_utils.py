"""
Sentinel-1/2 preprocessing utilities.

SAR (Sentinel-1 GRD) pipeline
-------------------------------
  linear intensity  →  clip(1e-10)  →  10·log10  →  clip[min_db, max_db]  →  uint8 [0,255]

EO (Sentinel-2 L2A) pipeline
-----------------------------
  uint16 reflectance (×10000)  →  /10000  →  clip[0, refl_cap]  →  /refl_cap  →  uint8 [0,255]

Both outputs are saved as uint8 PNGs; the model's normalise transform (mean=0.5, std=0.5)
maps [0,255] → [-1,1] via ToTensor then Normalize.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# SAR utilities
# ---------------------------------------------------------------------------


def linear_to_db(arr: NDArray[Any], eps: float = 1e-10) -> NDArray[Any]:
    """Convert linear power (intensity) to dB. Safe against zeros."""
    return 10.0 * np.log10(np.maximum(arr, eps))  # type: ignore[no-any-return]


def normalize_sar(
    arr: NDArray[Any],
    min_db: float = -25.0,
    max_db: float = 0.0,
    already_db: bool = False,
) -> NDArray[Any]:
    """
    Normalize a SAR channel to uint8 [0, 255].

    Args:
        arr:        2-D float array (single channel).
        min_db:     Lower clip value in dB (values below → 0).
        max_db:     Upper clip value in dB (values above → 255).
        already_db: If True, arr is already in dB and the linear→dB step is skipped.
    """
    db = arr if already_db else linear_to_db(arr)
    clipped = np.clip(db, min_db, max_db)
    normalized = (clipped - min_db) / (max_db - min_db)  # [0, 1]
    return (normalized * 255).astype(np.uint8)


def make_sar_image(
    bands: NDArray[Any],
    sar_channels: int = 1,
    channel_idx: int = 0,
    min_db: float = -25.0,
    max_db: float = 0.0,
    already_db: bool = False,
) -> NDArray[Any]:
    """
    Build an H×W (1-ch) or H×W×3 (3-ch) uint8 SAR image.

    Args:
        bands:        Array of shape (C, H, W) or (H, W, C).
        sar_channels: 1 → single polarization (VV), 3 → VV/VH/VV stacked to RGB.
        channel_idx:  Which channel to use as the primary (VV) polarization.
        min_db/max_db: dB clip range.
        already_db:   Skip linear→dB conversion if the data is already in dB.
    """
    if bands.ndim == 3 and bands.shape[0] <= 4:
        # (C, H, W) → (H, W, C)
        bands = np.transpose(bands, (1, 2, 0))

    vv = normalize_sar(bands[..., 0], min_db, max_db, already_db)

    if sar_channels == 1:
        return vv[..., np.newaxis]  # (H, W, 1)

    # 3-channel: VV, VH, VV  (common visualization for Sentinel-1)
    if bands.shape[-1] >= 2:
        vh = normalize_sar(bands[..., 1], min_db, max_db, already_db)
    else:
        vh = vv
    return np.stack([vv, vh, vv], axis=-1)  # (H, W, 3)


# ---------------------------------------------------------------------------
# EO (Sentinel-2) utilities
# ---------------------------------------------------------------------------

# Default RGB band indices within a standard Sentinel-2 13-band stack
# Band order in SEN12MS: B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12, B01
# RGB = B04 (index 2), B03 (index 1), B02 (index 0)  →  natural colour
S2_RGB_INDICES_SEN12MS = (2, 1, 0)  # B04, B03, B02 in SEN12MS band order
S2_RGB_INDICES_STANDARD = (3, 2, 1)  # B04, B03, B02 in standard ESA order (B01 first)


def normalize_eo(
    arr: NDArray[Any],
    reflectance_scale: float = 10_000.0,
    reflectance_cap: float = 0.3,
) -> NDArray[Any]:
    """
    Normalize a single EO channel to uint8 [0, 255].

    Args:
        arr:               2-D uint16 array (raw Sentinel-2 reflectance × 10000).
        reflectance_scale: Divide raw values by this to reach physical reflectance.
        reflectance_cap:   Clip reflectance at this value before scaling to [0,1].
                           0.3 covers most land surfaces; bright snow/cloud may saturate.
    """
    refl = arr.astype(np.float32) / reflectance_scale
    clipped = np.clip(refl, 0.0, reflectance_cap)
    normalized = clipped / reflectance_cap  # [0, 1]
    return (normalized * 255).astype(np.uint8)


def make_eo_image(
    bands: NDArray[Any],
    rgb_indices: tuple[int, int, int] = S2_RGB_INDICES_SEN12MS,
    reflectance_scale: float = 10_000.0,
    reflectance_cap: float = 0.3,
) -> NDArray[Any]:
    """
    Build an H×W×3 uint8 RGB EO image from a multi-band Sentinel-2 array.

    Args:
        bands:             (C, H, W) or (H, W, C) float/uint16 array.
        rgb_indices:       (R_idx, G_idx, B_idx) into the band axis.
        reflectance_scale: Scale factor for uint16 → physical reflectance.
        reflectance_cap:   Upper clip for reflectance before uint8 conversion.
    """
    if bands.ndim == 3 and bands.shape[0] <= 13:
        bands = np.transpose(bands, (1, 2, 0))  # (H, W, C)

    channels = [
        normalize_eo(bands[..., i], reflectance_scale, reflectance_cap) for i in rgb_indices
    ]
    return np.stack(channels, axis=-1)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Patch quality checks
# ---------------------------------------------------------------------------


def is_valid_patch(arr: NDArray[Any], min_valid_fraction: float = 0.8) -> bool:
    """
    Return True if the patch has enough valid (non-NaN, non-zero) pixels.

    Patches with heavy cloud cover or missing data are rejected.
    """
    valid = np.isfinite(arr) & (arr != 0)
    return float(valid.mean()) >= min_valid_fraction


# ---------------------------------------------------------------------------
# Side-by-side assembly
# ---------------------------------------------------------------------------


def make_side_by_side(sar_img: NDArray[Any], eo_img: NDArray[Any]) -> NDArray[Any]:
    """
    Concatenate SAR and EO images side-by-side into a [SAR | EO] uint8 image.

    Both inputs must have the same H×W. Single-channel SAR is broadcast to 3 channels
    so the combined image is always RGB (needed for standard PNG format).

    Returns: (H, 2W, 3) uint8 array.
    """
    h_sar, w_sar = sar_img.shape[:2]
    h_eo, w_eo = eo_img.shape[:2]
    if (h_sar, w_sar) != (h_eo, w_eo):
        raise ValueError(
            f"SAR {sar_img.shape[:2]} and EO {eo_img.shape[:2]} spatial dims must match"
        )

    # Broadcast 1-channel SAR to 3 channels for uniform PNG output
    if sar_img.ndim == 2:
        sar_img = np.stack([sar_img] * 3, axis=-1)
    elif sar_img.shape[-1] == 1:
        sar_img = np.concatenate([sar_img] * 3, axis=-1)

    if eo_img.ndim == 2:
        eo_img = np.stack([eo_img] * 3, axis=-1)

    return np.concatenate([sar_img, eo_img], axis=1)  # (H, 2W, 3)
