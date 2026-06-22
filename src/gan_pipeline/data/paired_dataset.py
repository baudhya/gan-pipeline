import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SideBySidePairedDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads SAR/EO pairs stored as a single [SAR | EO] image (standard pix2pix format).
    SAR is the left half, EO is the right half.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        sar_channels: int,
        eo_channels: int,
        augment: bool = True,
    ) -> None:
        root_path = Path(root) / split
        self.files = sorted(
            p
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
            for p in root_path.glob(ext)
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {root_path}")

        self.image_size = image_size
        self.sar_channels = sar_channels
        self.eo_channels = eo_channels
        self.augment = augment
        # Load slightly larger than target for random crop augmentation
        self.load_size = int(image_size * 1.12)

    def __len__(self) -> int:
        return len(self.files)

    def _to_tensor_normalized(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)  # [0, 1]
        ch = t.shape[0]
        return TF.normalize(t, [0.5] * ch, [0.5] * ch)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full = Image.open(self.files[idx])
        w, h = full.size
        half = w // 2

        sar_pil = full.crop((0, 0, half, h))
        eo_pil = full.crop((half, 0, w, h))

        sar_pil = sar_pil.convert("L" if self.sar_channels == 1 else "RGB")
        eo_pil = eo_pil.convert("L" if self.eo_channels == 1 else "RGB")

        if self.augment:
            sar_pil = TF.resize(
                sar_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )

            # Synchronized random crop
            i, j, th, tw = transforms.RandomCrop.get_params(
                sar_pil, (self.image_size, self.image_size)
            )
            sar_pil = TF.crop(sar_pil, i, j, th, tw)
            eo_pil = TF.crop(eo_pil, i, j, th, tw)

            # Synchronized random horizontal flip
            if random.random() > 0.5:
                sar_pil = TF.hflip(sar_pil)
                eo_pil = TF.hflip(eo_pil)
        else:
            sar_pil = TF.resize(
                sar_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )

        return {
            "sar": self._to_tensor_normalized(sar_pil),
            "eo": self._to_tensor_normalized(eo_pil),
        }


class SeparateDirPairedDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads SAR/EO pairs from two separate directories (trainA/ = SAR, trainB/ = EO).
    Filenames must match across both directories.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        sar_channels: int,
        eo_channels: int,
        augment: bool = True,
        sar_dir: str = "A",
        eo_dir: str = "B",
    ) -> None:
        sar_root = Path(root) / f"{split}{sar_dir}"
        eo_root = Path(root) / f"{split}{eo_dir}"

        self.sar_files = sorted(sar_root.iterdir())
        self.eo_files = sorted(eo_root.iterdir())

        if len(self.sar_files) != len(self.eo_files):
            raise ValueError(
                f"SAR/EO file count mismatch: {len(self.sar_files)} vs {len(self.eo_files)}"
            )

        self.image_size = image_size
        self.sar_channels = sar_channels
        self.eo_channels = eo_channels
        self.augment = augment
        self.load_size = int(image_size * 1.12)

    def __len__(self) -> int:
        return len(self.sar_files)

    def _to_tensor_normalized(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)
        ch = t.shape[0]
        return TF.normalize(t, [0.5] * ch, [0.5] * ch)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sar_pil = Image.open(self.sar_files[idx]).convert("L" if self.sar_channels == 1 else "RGB")
        eo_pil = Image.open(self.eo_files[idx]).convert("L" if self.eo_channels == 1 else "RGB")

        if self.augment:
            sar_pil = TF.resize(
                sar_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )

            i, j, th, tw = transforms.RandomCrop.get_params(
                sar_pil, (self.image_size, self.image_size)
            )
            sar_pil = TF.crop(sar_pil, i, j, th, tw)
            eo_pil = TF.crop(eo_pil, i, j, th, tw)

            if random.random() > 0.5:
                sar_pil = TF.hflip(sar_pil)
                eo_pil = TF.hflip(eo_pil)
        else:
            sar_pil = TF.resize(
                sar_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )

        return {
            "sar": self._to_tensor_normalized(sar_pil),
            "eo": self._to_tensor_normalized(eo_pil),
        }


class SentinelS1S2Dataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads SAR/EO pairs from a multi-category Sentinel dataset with structure:
        {root}/{category}/s1/*.png  (SAR — Sentinel-1)
        {root}/{category}/s2/*.png  (EO  — Sentinel-2)

    Each s1 file is paired with its s2 counterpart by substituting '_s1_' → '_s2_'
    in the filename (e.g. ROIs1970_fall_s1_59_p001.png ↔ ROIs1970_fall_s2_59_p001.png).
    A deterministic 90/10 train/val split is applied with a fixed seed.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        sar_channels: int,
        eo_channels: int,
        augment: bool = True,
    ) -> None:
        root_path = Path(root)
        all_s1: list[Path] = sorted(
            p
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
            for p in root_path.rglob(f"s1/{ext}")
        )
        pairs: list[tuple[Path, Path]] = []
        for s1_path in all_s1:
            s2_name = s1_path.name.replace("_s1_", "_s2_")
            s2_path = s1_path.parent.parent / "s2" / s2_name
            if s2_path.exists():
                pairs.append((s1_path, s2_path))

        if not pairs:
            raise FileNotFoundError(
                f"No matched SAR/EO pairs found under {root_path}. "
                "Expected structure: {root}/{category}/s1/ and {root}/{category}/s2/ "
                "with filenames like ROI_s1_tile_patch.png ↔ ROI_s2_tile_patch.png."
            )

        rng = random.Random(42)
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.1))
        if split == "train":
            self.pairs = shuffled[n_val:]
        elif split in ("val", "test"):
            self.pairs = shuffled[:n_val]
        else:
            raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val', or 'test'.")

        if not self.pairs:
            raise FileNotFoundError(f"No pairs available for split {split!r} in {root_path}")

        self.image_size = image_size
        self.sar_channels = sar_channels
        self.eo_channels = eo_channels
        self.augment = augment
        self.load_size = int(image_size * 1.12)

    def __len__(self) -> int:
        return len(self.pairs)

    def _to_tensor_normalized(self, img: Image.Image) -> torch.Tensor:
        t = TF.to_tensor(img)
        ch = t.shape[0]
        return TF.normalize(t, [0.5] * ch, [0.5] * ch)  # type: ignore[no-any-return]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s1_path, s2_path = self.pairs[idx]
        sar_pil = Image.open(s1_path).convert("L" if self.sar_channels == 1 else "RGB")
        eo_pil = Image.open(s2_path).convert("L" if self.eo_channels == 1 else "RGB")

        if self.augment:
            sar_pil = TF.resize(
                sar_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.load_size, self.load_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            i, j, th, tw = transforms.RandomCrop.get_params(
                sar_pil, (self.image_size, self.image_size)
            )
            sar_pil = TF.crop(sar_pil, i, j, th, tw)
            eo_pil = TF.crop(eo_pil, i, j, th, tw)
            if random.random() > 0.5:
                sar_pil = TF.hflip(sar_pil)
                eo_pil = TF.hflip(eo_pil)
        else:
            sar_pil = TF.resize(
                sar_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            eo_pil = TF.resize(
                eo_pil,
                [self.image_size, self.image_size],
                interpolation=TF.InterpolationMode.BICUBIC,
            )

        return {
            "sar": self._to_tensor_normalized(sar_pil),
            "eo": self._to_tensor_normalized(eo_pil),
        }


def get_paired_dataloader(
    root: str,
    split: str,
    image_size: int,
    sar_channels: int,
    eo_channels: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    dataset_format: str = "side_by_side",
) -> DataLoader:  # type: ignore[type-arg]
    dataset: Dataset[dict[str, torch.Tensor]]
    if dataset_format == "side_by_side":
        dataset = SideBySidePairedDataset(
            root, split, image_size, sar_channels, eo_channels, augment
        )
    elif dataset_format == "separate_dirs":
        dataset = SeparateDirPairedDataset(
            root, split, image_size, sar_channels, eo_channels, augment
        )
    elif dataset_format == "sentinel_s1s2":
        dataset = SentinelS1S2Dataset(root, split, image_size, sar_channels, eo_channels, augment)
    else:
        raise ValueError(f"Unknown dataset_format: {dataset_format}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
