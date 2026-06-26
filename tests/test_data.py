from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gan_pipeline.data.paired_dataset import (
    SentinelS1S2Dataset,
    SeparateDirPairedDataset,
    SideBySidePairedDataset,
    get_paired_dataloader,
)
from gan_pipeline.data.transforms import train_transform, val_transform
from gan_pipeline.utils import setup_logging

# --- Paired transforms ---


def test_val_transform_shape_and_range() -> None:
    t = val_transform(64)
    sar = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    eo = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    sar_out, eo_out = t(sar, eo)
    assert sar_out.size == (64, 64)
    assert eo_out.size == (64, 64)


@pytest.mark.parametrize("size", [32, 64, 128])
def test_val_transform_output_size(size: int) -> None:
    t = val_transform(size)
    sar = Image.fromarray(np.random.randint(0, 255, (200, 150), dtype=np.uint8))
    eo = Image.fromarray(np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8))
    sar_out, eo_out = t(sar, eo)
    assert sar_out.size == (size, size)
    assert eo_out.size == (size, size)


def test_train_transform_output_size() -> None:
    t = train_transform(64)
    sar = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
    eo = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    sar_out, eo_out = t(sar, eo)
    assert sar_out.size == (64, 64)
    assert eo_out.size == (64, 64)


def test_train_transform_synchronized() -> None:
    """Crop and flip must be identical for both images.

    Pass identical SAR and EO images — if the same transform is applied to
    both, the outputs must match pixel-for-pixel.
    """
    t = train_transform(64)
    data = np.arange(100 * 100, dtype=np.uint8).reshape(100, 100)
    sar = Image.fromarray(data)
    eo = Image.fromarray(data)  # identical to sar
    sar_out, eo_out = t(sar, eo)
    assert list(sar_out.getdata()) == list(eo_out.getdata())


# --- SideBySidePairedDataset ---


def test_side_by_side_no_files_raises(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError):
        SideBySidePairedDataset(str(tmp_path), "train", 1, 3)


# --- SeparateDirPairedDataset ---


def _make_separate_dir(tmp_path: Path, n: int = 3) -> Path:
    for sub in ["trainA", "trainB"]:
        d = tmp_path / sub
        d.mkdir()
        for i in range(n):
            img = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
            img.save(d / f"{i:04d}.png")
    return tmp_path


def test_separate_dir_dataset_augmented(tmp_path: Path) -> None:
    root = _make_separate_dir(tmp_path)
    ds = SeparateDirPairedDataset(str(root), "train", 1, 3, transform=train_transform(64))
    assert len(ds) == 3
    sample = ds[0]
    assert sample["sar"].shape == (1, 64, 64)
    assert sample["eo"].shape == (3, 64, 64)
    assert sample["sar"].min() >= -1.0 and sample["sar"].max() <= 1.0


def test_separate_dir_dataset_no_augment(tmp_path: Path) -> None:
    root = _make_separate_dir(tmp_path)
    ds = SeparateDirPairedDataset(str(root), "train", 1, 3, transform=val_transform(64))
    sample = ds[0]
    assert sample["sar"].shape == (1, 64, 64)
    assert sample["eo"].shape == (3, 64, 64)


def test_separate_dir_dataset_mismatch_raises(tmp_path: Path) -> None:
    (tmp_path / "trainA").mkdir()
    (tmp_path / "trainB").mkdir()
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(tmp_path / "trainA" / "img.png")
    with pytest.raises(ValueError, match="mismatch"):
        SeparateDirPairedDataset(str(tmp_path), "train", 1, 3)


# --- get_paired_dataloader ---


def test_get_paired_dataloader_separate_dirs(tmp_path: Path) -> None:
    root = _make_separate_dir(tmp_path)
    loader = get_paired_dataloader(
        str(root),
        "train",
        sar_channels=1,
        eo_channels=3,
        batch_size=2,
        image_size=64,
        num_workers=0,
        dataset_format="separate_dirs",
    )
    batch = next(iter(loader))
    assert "sar" in batch and "eo" in batch


def test_get_paired_dataloader_custom_transform(tmp_path: Path) -> None:
    """Injected transform overrides the default augment-based selection."""
    root = _make_separate_dir(tmp_path)

    calls: list[str] = []

    def _custom(sar: Image.Image, eo: Image.Image) -> tuple[Image.Image, Image.Image]:
        calls.append("called")
        return val_transform(64)(sar, eo)

    loader = get_paired_dataloader(
        str(root),
        "train",
        sar_channels=1,
        eo_channels=3,
        batch_size=2,
        image_size=64,
        num_workers=0,
        dataset_format="separate_dirs",
        transform=_custom,
    )
    next(iter(loader))
    assert len(calls) > 0


def test_get_paired_dataloader_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset_format"):
        get_paired_dataloader(
            str(tmp_path),
            "train",
            sar_channels=1,
            eo_channels=3,
            batch_size=2,
            dataset_format="xyz",
        )


# --- SentinelS1S2Dataset ---


def _make_sentinel_dir(tmp_path: Path, n_per_category: int = 4) -> Path:
    """Create a minimal Sentinel-style multi-category dataset."""
    categories = ["agri", "urban"]
    for cat in categories:
        for sensor in ("s1", "s2"):
            (tmp_path / cat / sensor).mkdir(parents=True)
        for i in range(n_per_category):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
            img.save(tmp_path / cat / "s1" / f"ROIs1970_fall_s1_59_p{i:04d}.png")
            img.save(tmp_path / cat / "s2" / f"ROIs1970_fall_s2_59_p{i:04d}.png")
    return tmp_path


def test_sentinel_s1s2_dataset_shape(tmp_path: Path) -> None:
    _make_sentinel_dir(tmp_path, n_per_category=10)
    ds = SentinelS1S2Dataset(str(tmp_path), "train", 1, 3, transform=val_transform(64))
    sample = ds[0]
    assert sample["sar"].shape == (1, 64, 64)
    assert sample["eo"].shape == (3, 64, 64)
    assert sample["sar"].min() >= -1.0 and sample["sar"].max() <= 1.0


def test_sentinel_s1s2_train_val_split(tmp_path: Path) -> None:
    _make_sentinel_dir(tmp_path, n_per_category=10)  # 20 total pairs
    train_ds = SentinelS1S2Dataset(str(tmp_path), "train", 1, 3)
    val_ds = SentinelS1S2Dataset(str(tmp_path), "val", 1, 3)
    assert len(train_ds) + len(val_ds) == 20
    assert len(val_ds) >= 1
    assert len(train_ds) > len(val_ds)


def test_sentinel_s1s2_no_pairs_raises(tmp_path: Path) -> None:
    (tmp_path / "cat" / "s1").mkdir(parents=True)
    (tmp_path / "cat" / "s2").mkdir(parents=True)
    img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
    img.save(tmp_path / "cat" / "s1" / "ROIs1970_fall_s1_59_p001.png")
    # s2 counterpart missing → no pairs
    with pytest.raises(FileNotFoundError):
        SentinelS1S2Dataset(str(tmp_path), "train", 1, 3)


def test_get_paired_dataloader_sentinel(tmp_path: Path) -> None:
    _make_sentinel_dir(tmp_path, n_per_category=6)
    loader = get_paired_dataloader(
        str(tmp_path),
        "train",
        sar_channels=1,
        eo_channels=3,
        batch_size=2,
        image_size=64,
        num_workers=0,
        dataset_format="sentinel_s1s2",
    )
    batch = next(iter(loader))
    assert batch["sar"].shape == (2, 1, 64, 64)
    assert batch["eo"].shape == (2, 3, 64, 64)


# --- setup_logging ---


def test_setup_logging(tmp_path: Path) -> None:
    setup_logging(tmp_path)  # just verify it doesn't raise
