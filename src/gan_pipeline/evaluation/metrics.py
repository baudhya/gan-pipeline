from pathlib import Path

import torch
from loguru import logger


def compute_fid(real_path: Path, fake_path: Path, device: torch.device) -> float:
    try:
        from torch_fidelity import calculate_metrics  # type: ignore[import-untyped]

        metrics = calculate_metrics(
            input1=str(real_path),
            input2=str(fake_path),
            cuda=device.type == "cuda",
            fid=True,
            verbose=False,
        )
        return float(metrics["frechet_inception_distance"])
    except ImportError:
        logger.warning("torch-fidelity not installed. Run: pip install torch-fidelity")
        return float("nan")


def compute_inception_score(
    fake_path: Path, device: torch.device, splits: int = 10
) -> tuple[float, float]:
    try:
        from torch_fidelity import calculate_metrics  # type: ignore[import-untyped]

        metrics = calculate_metrics(
            input1=str(fake_path),
            cuda=device.type == "cuda",
            isc=True,
            isc_splits=splits,
            verbose=False,
        )
        return float(metrics["inception_score_mean"]), float(metrics["inception_score_std"])
    except ImportError:
        logger.warning("torch-fidelity not installed. Run: pip install torch-fidelity")
        return float("nan"), float("nan")
