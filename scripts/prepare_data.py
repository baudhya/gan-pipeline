#!/usr/bin/env python3
"""
Sentinel-1/2 data preparation for SAR→EO paired image translation.

Two operating modes
-------------------
sen12ms  — Process the SEN12MS benchmark dataset (pre-paired 256×256 GeoTIFF patches).
            Download from: https://mediatum.ub.tum.de/1474000

scenes   — Chip co-registered Sentinel-1 and Sentinel-2 GeoTIFF scenes into patches.
            Both scenes must cover the same geographic extent and be in the same CRS.
            Use ESA SNAP or GDAL to coregister before running this script.

Output format
-------------
Side-by-side [SAR | EO] PNG images consumed directly by the pix2pix training pipeline:

  <output_dir>/
    train/00001.png   # 512×256 RGB: left half = SAR, right half = EO
    val/00001.png
    test/00001.png

SAR preprocessing
-----------------
  1. Read Sentinel-1 GRD backscatter (float32, linear power scale by default).
  2. Convert to dB:  10·log₁₀(intensity),  clip to [sar_min_db, sar_max_db].
  3. Map to uint8 [0, 255].

EO preprocessing
----------------
  1. Read Sentinel-2 L2A BOA reflectance (uint16, ×10000 scale).
  2. Select RGB bands (B04=Red, B03=Green, B02=Blue by default).
  3. Clip reflectance to [0, refl_cap] and map to uint8 [0, 255].

Usage examples
--------------
# SEN12MS dataset
python scripts/prepare_data.py \\
  --mode sen12ms \\
  --s1-dir /data/SEN12MS/s1 \\
  --s2-dir /data/SEN12MS/s2 \\
  --output-dir data/sar_eo \\
  --sar-channels 1 \\
  --val-split 0.1 \\
  --test-split 0.1

# Raw co-registered scenes
python scripts/prepare_data.py \\
  --mode scenes \\
  --s1-dir /data/raw/sentinel1 \\
  --s2-dir /data/raw/sentinel2 \\
  --output-dir data/sar_eo \\
  --image-size 256 \\
  --stride 128 \\
  --min-valid-fraction 0.9

# Already-in-dB SAR data (SEN12MS stores S1 in dB by default)
python scripts/prepare_data.py \\
  --mode sen12ms \\
  --s1-dir /data/SEN12MS/s1 \\
  --s2-dir /data/SEN12MS/s2 \\
  --output-dir data/sar_eo \\
  --sar-already-db
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from gan_pipeline.data.sentinel_utils import (
    is_valid_patch,
    make_eo_image,
    make_sar_image,
    make_side_by_side,
    S2_RGB_INDICES_SEN12MS,
    S2_RGB_INDICES_STANDARD,
)

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_rasterio() -> None:
    if not HAS_RASTERIO:
        logger.error("rasterio is required. Install with:  pip install rasterio")
        sys.exit(1)


def _read_geotiff(path: Path) -> np.ndarray:
    """Read all bands from a GeoTIFF. Returns (C, H, W) float32 array."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        # Replace nodata with NaN
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _read_geotiff_window(src: "rasterio.DatasetReader", window: Window) -> np.ndarray:
    """Read a spatial window from an open rasterio dataset. Returns (C, H, W)."""
    arr = src.read(window=window).astype(np.float32)
    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan
    return arr


def _save_pair(sar_img: np.ndarray, eo_img: np.ndarray, dest: Path) -> None:
    """Assemble a side-by-side image and save as PNG."""
    combined = make_side_by_side(sar_img, eo_img)
    Image.fromarray(combined).save(dest)


def _split_indices(n: int, val_split: float, test_split: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Random train/val/test split of [0, n)."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))
    test = indices[:n_test]
    val = indices[n_test:n_test + n_val]
    train = indices[n_test + n_val:]
    return train, val, test


def _make_split_dirs(output_dir: Path) -> dict[str, Path]:
    splits = {}
    for name in ("train", "val", "test"):
        p = output_dir / name
        p.mkdir(parents=True, exist_ok=True)
        splits[name] = p
    return splits


# ---------------------------------------------------------------------------
# Mode: SEN12MS
# ---------------------------------------------------------------------------

def _discover_sen12ms_pairs(s1_root: Path, s2_root: Path) -> list[tuple[Path, Path]]:
    """
    Discover matching (s1_patch, s2_patch) GeoTIFF pairs from a SEN12MS directory tree.

    SEN12MS layout:
      s1/ROIs{id}_{season}_s1_{scene}/s1_{patch}.tif
      s2/ROIs{id}_{season}_s2_{scene}/s2_{patch}.tif

    Pairing is done by matching (roi_id, season, scene, patch_id) across s1 and s2 roots.
    """
    s1_files: dict[tuple[str, ...], Path] = {}
    for f in sorted(s1_root.rglob("*.tif")):
        # e.g. ROIs1158_spring_s1_1/s1_1.tif
        # key: (roi_season, scene_id, patch_id)
        parts = f.parts
        scene_dir = parts[-2]          # e.g. ROIs1158_spring_s1_1
        patch_stem = f.stem            # e.g. s1_1
        patch_id = patch_stem.split("_")[-1]
        # Normalise scene dir: drop the "s1" sensor tag for matching
        scene_key = scene_dir.replace("_s1_", "_").replace("_s2_", "_")
        s1_files[(scene_key, patch_id)] = f

    pairs: list[tuple[Path, Path]] = []
    for f in sorted(s2_root.rglob("*.tif")):
        parts = f.parts
        scene_dir = parts[-2]
        patch_stem = f.stem
        patch_id = patch_stem.split("_")[-1]
        scene_key = scene_dir.replace("_s1_", "_").replace("_s2_", "_")
        key = (scene_key, patch_id)
        if key in s1_files:
            pairs.append((s1_files[key], f))

    return pairs


def process_sen12ms(args: argparse.Namespace) -> None:
    _require_rasterio()

    s1_root = Path(args.s1_dir)
    s2_root = Path(args.s2_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Discovering SEN12MS pairs in:\n  S1: {s1_root}\n  S2: {s2_root}")
    pairs = _discover_sen12ms_pairs(s1_root, s2_root)
    if not pairs:
        logger.error("No matching pairs found. Check --s1-dir / --s2-dir paths.")
        sys.exit(1)
    logger.info(f"Found {len(pairs):,} paired patches")

    rgb_indices = S2_RGB_INDICES_SEN12MS if not args.s2_standard_order else S2_RGB_INDICES_STANDARD
    splits_dirs = _make_split_dirs(output_dir)
    train_idx, val_idx, test_idx = _split_indices(len(pairs), args.val_split, args.test_split, args.seed)
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})
    split_map.update({i: "test" for i in test_idx})

    saved = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for idx, (s1_path, s2_path) in enumerate(tqdm(pairs, desc="Processing patches")):
        try:
            s1_bands = _read_geotiff(s1_path)   # (C, H, W) float32
            s2_bands = _read_geotiff(s2_path)

            # Quality check on both arrays
            if not is_valid_patch(s1_bands, args.min_valid_fraction):
                skipped += 1
                continue
            if not is_valid_patch(s2_bands, args.min_valid_fraction):
                skipped += 1
                continue

            sar_img = make_sar_image(
                s1_bands,
                sar_channels=args.sar_channels,
                min_db=args.sar_min_db,
                max_db=args.sar_max_db,
                already_db=args.sar_already_db,
            )
            eo_img = make_eo_image(
                s2_bands,
                rgb_indices=rgb_indices,
                reflectance_scale=args.s2_scale,
                reflectance_cap=args.refl_cap,
            )

            split = split_map[idx]
            dest = splits_dirs[split] / f"{idx:06d}.png"
            _save_pair(sar_img, eo_img, dest)
            saved[split] += 1

        except Exception as exc:
            logger.warning(f"Skipping {s1_path.name}: {exc}")
            skipped += 1

    logger.info(
        f"Done. Saved — train: {saved['train']:,}  val: {saved['val']:,}  "
        f"test: {saved['test']:,}  skipped: {skipped:,}"
    )
    logger.info(f"Output: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# Mode: co-registered scenes
# ---------------------------------------------------------------------------

def _iter_scene_pairs(s1_root: Path, s2_root: Path) -> list[tuple[Path, Path]]:
    """
    Match S1 and S2 scenes by filename stem.

    Supported layouts:
      a) Flat:  s1_root/scene_001.tif  ↔  s2_root/scene_001.tif
      b) Subdir: s1_root/scene_001/sentinel1.tif  ↔  s2_root/scene_001/sentinel2.tif
    """
    s1_flat = {f.stem: f for f in sorted(s1_root.glob("*.tif"))}
    s2_flat = {f.stem: f for f in sorted(s2_root.glob("*.tif"))}

    pairs: list[tuple[Path, Path]] = []

    # Flat layout: match by stem
    for stem, s1f in s1_flat.items():
        if stem in s2_flat:
            pairs.append((s1f, s2_flat[stem]))

    # Subdir layout: s1_root/<scene>/*.tif  ↔  s2_root/<scene>/*.tif
    if not pairs:
        for scene_dir in sorted(s1_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            s1_files = list(scene_dir.glob("*.tif"))
            s2_scene = s2_root / scene_dir.name
            if not s2_scene.is_dir():
                continue
            s2_files = list(s2_scene.glob("*.tif"))
            if s1_files and s2_files:
                pairs.append((s1_files[0], s2_files[0]))

    return pairs


def process_scenes(args: argparse.Namespace) -> None:
    _require_rasterio()

    s1_root = Path(args.s1_dir)
    s2_root = Path(args.s2_dir)
    output_dir = Path(args.output_dir)
    image_size = args.image_size
    stride = args.stride or image_size

    scene_pairs = _iter_scene_pairs(s1_root, s2_root)
    if not scene_pairs:
        logger.error("No matching scene pairs found. Check --s1-dir / --s2-dir paths.")
        sys.exit(1)
    logger.info(f"Found {len(scene_pairs)} scene pair(s)")

    rgb_indices = S2_RGB_INDICES_SEN12MS if not args.s2_standard_order else S2_RGB_INDICES_STANDARD
    splits_dirs = _make_split_dirs(output_dir)

    all_patches: list[tuple[np.ndarray, np.ndarray]] = []

    for s1_path, s2_path in tqdm(scene_pairs, desc="Chipping scenes"):
        logger.info(f"  S1: {s1_path.name}  S2: {s2_path.name}")
        try:
            with rasterio.open(s1_path) as s1_src, rasterio.open(s2_path) as s2_src:
                h = min(s1_src.height, s2_src.height)
                w = min(s1_src.width, s2_src.width)

                for row_off in range(0, h - image_size + 1, stride):
                    for col_off in range(0, w - image_size + 1, stride):
                        win = Window(col_off, row_off, image_size, image_size)
                        s1_patch = _read_geotiff_window(s1_src, win)
                        s2_patch = _read_geotiff_window(s2_src, win)

                        if not is_valid_patch(s1_patch, args.min_valid_fraction):
                            continue
                        if not is_valid_patch(s2_patch, args.min_valid_fraction):
                            continue

                        all_patches.append((s1_patch, s2_patch))

        except Exception as exc:
            logger.warning(f"Error processing scene pair ({s1_path.name}, {s2_path.name}): {exc}")

    if not all_patches:
        logger.error("No valid patches extracted. Lower --min-valid-fraction or check your data.")
        sys.exit(1)

    logger.info(f"Extracted {len(all_patches):,} valid patches — splitting and saving...")

    train_idx, val_idx, test_idx = _split_indices(len(all_patches), args.val_split, args.test_split, args.seed)
    split_map = {i: "train" for i in train_idx}
    split_map.update({i: "val" for i in val_idx})
    split_map.update({i: "test" for i in test_idx})

    saved = {"train": 0, "val": 0, "test": 0}

    for idx, (s1_patch, s2_patch) in enumerate(tqdm(all_patches, desc="Saving")):
        sar_img = make_sar_image(
            s1_patch,
            sar_channels=args.sar_channels,
            min_db=args.sar_min_db,
            max_db=args.sar_max_db,
            already_db=args.sar_already_db,
        )
        eo_img = make_eo_image(
            s2_patch,
            rgb_indices=rgb_indices,
            reflectance_scale=args.s2_scale,
            reflectance_cap=args.refl_cap,
        )
        split = split_map[idx]
        dest = splits_dirs[split] / f"{idx:06d}.png"
        _save_pair(sar_img, eo_img, dest)
        saved[split] += 1

    logger.info(
        f"Done. Saved — train: {saved['train']:,}  val: {saved['val']:,}  "
        f"test: {saved['test']:,}"
    )
    logger.info(f"Output: {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode", required=True, choices=["sen12ms", "scenes"],
        help="Processing mode (see module docstring for details)",
    )
    p.add_argument("--s1-dir", required=True, metavar="PATH",
                   help="Root directory containing Sentinel-1 GeoTIFFs")
    p.add_argument("--s2-dir", required=True, metavar="PATH",
                   help="Root directory containing Sentinel-2 GeoTIFFs")
    p.add_argument("--output-dir", default="data/sar_eo", metavar="PATH",
                   help="Destination for train/val/test PNG files (default: data/sar_eo)")

    # SAR options
    sar = p.add_argument_group("SAR (Sentinel-1) options")
    sar.add_argument("--sar-channels", type=int, default=1, choices=[1, 3],
                     help="1 = VV only (grayscale); 3 = VV/VH/VV stacked (default: 1)")
    sar.add_argument("--sar-min-db", type=float, default=-25.0,
                     help="Lower dB clip value — pixels below this → 0 (default: -25)")
    sar.add_argument("--sar-max-db", type=float, default=0.0,
                     help="Upper dB clip value — pixels above this → 255 (default: 0)")
    sar.add_argument("--sar-already-db", action="store_true",
                     help="Skip linear→dB conversion (use if data is already in dB, e.g. SEN12MS)")

    # EO options
    eo = p.add_argument_group("EO (Sentinel-2) options")
    eo.add_argument("--s2-scale", type=float, default=10_000.0,
                    help="Divide raw uint16 values by this to get physical reflectance (default: 10000)")
    eo.add_argument("--refl-cap", type=float, default=0.3,
                    help="Clip reflectance at this value before uint8 scaling (default: 0.3)")
    eo.add_argument("--s2-standard-order", action="store_true",
                    help="Use standard ESA band order (B01 first) instead of SEN12MS order (B02 first)")

    # Patch / scene options
    patch = p.add_argument_group("Patch extraction options (scenes mode)")
    patch.add_argument("--image-size", type=int, default=256,
                       help="Output patch size in pixels (default: 256)")
    patch.add_argument("--stride", type=int, default=None,
                       help="Sliding window stride (default: image-size, i.e. no overlap)")
    patch.add_argument("--min-valid-fraction", type=float, default=0.8,
                       help="Minimum fraction of finite, non-zero pixels for a patch to be kept (default: 0.8)")

    # Split options
    split = p.add_argument_group("Train/val/test split")
    split.add_argument("--val-split", type=float, default=0.1,
                       help="Fraction of patches for validation (default: 0.1)")
    split.add_argument("--test-split", type=float, default=0.1,
                       help="Fraction of patches for test (default: 0.1)")
    split.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    if args.mode == "sen12ms":
        process_sen12ms(args)
    else:
        process_scenes(args)


if __name__ == "__main__":
    main()
