"""Generate synthetic side-by-side SAR/EO PNGs for smoke-testing the training loop."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def make_pair(rng: random.Random, image_size: int) -> Image.Image:
    """Grayscale noise for SAR (left) + coloured noise for EO (right)."""
    hw = image_size
    sar = np.array([rng.randint(0, 255) for _ in range(hw * hw)], dtype=np.uint8).reshape(hw, hw)
    sar_rgb = np.stack([sar, sar, sar], axis=-1)
    eo = np.array([rng.randint(0, 255) for _ in range(hw * hw * 3)], dtype=np.uint8).reshape(
        hw, hw, 3
    )
    side_by_side = np.concatenate([sar_rgb, eo], axis=1)  # (H, 2W, 3)
    return Image.fromarray(side_by_side)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sar_eo")
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.output_dir)

    splits = {"train": args.train, "val": args.val}
    for split, n in splits.items():
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = make_pair(rng, args.image_size)
            img.save(out / f"{i:06d}.png")
        print(f"  {split}: {n} images → {out}")

    print(f"\nDone. Dataset at {root}")


if __name__ == "__main__":
    main()
