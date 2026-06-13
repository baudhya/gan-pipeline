"""Tests for Sentinel-1/2 preprocessing utilities."""
import numpy as np
import pytest

from gan_pipeline.data.sentinel_utils import (
    is_valid_patch,
    linear_to_db,
    make_eo_image,
    make_sar_image,
    make_side_by_side,
    normalize_eo,
    normalize_sar,
)


# --- linear_to_db ---

def test_linear_to_db_basic() -> None:
    arr = np.array([1.0, 10.0, 100.0])
    db = linear_to_db(arr)
    np.testing.assert_allclose(db, [0.0, 10.0, 20.0], atol=1e-5)


def test_linear_to_db_zero_safe() -> None:
    arr = np.array([0.0, -1.0])
    db = linear_to_db(arr)
    assert np.all(np.isfinite(db))


# --- normalize_sar ---

def test_normalize_sar_range() -> None:
    arr = np.array([-25.0, -12.5, 0.0])
    out = normalize_sar(arr, min_db=-25.0, max_db=0.0, already_db=True)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [0, 127, 255])


def test_normalize_sar_clips_outside_range() -> None:
    arr = np.array([-50.0, 50.0])
    out = normalize_sar(arr, min_db=-25.0, max_db=0.0, already_db=True)
    assert out[0] == 0
    assert out[1] == 255


def test_normalize_sar_linear_input() -> None:
    # linear 1.0 → 0 dB → should map to 255 with default range [-25, 0]
    arr = np.array([1.0])
    out = normalize_sar(arr, min_db=-25.0, max_db=0.0, already_db=False)
    assert out[0] == 255


# --- normalize_eo ---

def test_normalize_eo_range() -> None:
    arr = np.array([0, 1500, 3000], dtype=np.uint16)   # 3000/10000 = 0.3 = cap → 255
    out = normalize_eo(arr, reflectance_scale=10_000.0, reflectance_cap=0.3)
    assert out.dtype == np.uint8
    assert out[0] == 0
    assert out[-1] == 255


def test_normalize_eo_clips_bright_pixels() -> None:
    arr = np.array([9999, 10000], dtype=np.uint16)
    out = normalize_eo(arr, reflectance_scale=10_000.0, reflectance_cap=0.3)
    assert out[0] == 255 and out[1] == 255


# --- make_sar_image ---

@pytest.mark.parametrize("sar_channels", [1, 3])
def test_make_sar_image_channels(sar_channels: int) -> None:
    bands = np.random.rand(2, 64, 64).astype(np.float32)   # (C, H, W) linear scale
    out = make_sar_image(bands, sar_channels=sar_channels)
    assert out.dtype == np.uint8
    expected_ch = 1 if sar_channels == 1 else 3
    assert out.shape == (64, 64, expected_ch)


def test_make_sar_image_hwc_input() -> None:
    bands = np.random.rand(64, 64, 2).astype(np.float32)   # (H, W, C)
    out = make_sar_image(bands, sar_channels=1)
    assert out.shape == (64, 64, 1)


# --- make_eo_image ---

def test_make_eo_image_shape() -> None:
    bands = np.random.randint(0, 3000, (13, 64, 64), dtype=np.uint16)
    out = make_eo_image(bands)
    assert out.dtype == np.uint8
    assert out.shape == (64, 64, 3)


def test_make_eo_image_hwc_input() -> None:
    bands = np.random.randint(0, 3000, (64, 64, 13), dtype=np.uint16)
    out = make_eo_image(bands)
    assert out.shape == (64, 64, 3)


# --- is_valid_patch ---

def test_valid_patch_all_finite() -> None:
    arr = np.ones((64, 64), dtype=np.float32)
    assert is_valid_patch(arr, min_valid_fraction=0.8)


def test_invalid_patch_mostly_nan() -> None:
    arr = np.full((64, 64), np.nan, dtype=np.float32)
    arr[:5, :5] = 1.0   # only a small corner is valid
    assert not is_valid_patch(arr, min_valid_fraction=0.8)


def test_invalid_patch_mostly_zero() -> None:
    arr = np.zeros((64, 64), dtype=np.float32)
    arr[0, 0] = 1.0
    assert not is_valid_patch(arr, min_valid_fraction=0.8)


# --- make_side_by_side ---

def test_side_by_side_shape() -> None:
    sar = np.zeros((64, 64, 1), dtype=np.uint8)
    eo = np.zeros((64, 64, 3), dtype=np.uint8)
    out = make_side_by_side(sar, eo)
    assert out.shape == (64, 128, 3)   # width doubles, single-ch SAR → 3ch


def test_side_by_side_both_3ch() -> None:
    sar = np.zeros((64, 64, 3), dtype=np.uint8)
    eo = np.zeros((64, 64, 3), dtype=np.uint8)
    out = make_side_by_side(sar, eo)
    assert out.shape == (64, 128, 3)


def test_side_by_side_spatial_mismatch_raises() -> None:
    sar = np.zeros((64, 64, 1), dtype=np.uint8)
    eo = np.zeros((32, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="spatial dims must match"):
        make_side_by_side(sar, eo)


def test_side_by_side_sar_on_left() -> None:
    sar = np.full((4, 4, 1), 10, dtype=np.uint8)
    eo = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = make_side_by_side(sar, eo)
    # Left half should be the SAR value
    assert out[0, 0, 0] == 10
    # Right half should be the EO value
    assert out[0, 4, 0] == 200
