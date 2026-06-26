"""Tests for inference (load_generator) and evaluation (metrics) modules."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from gan_pipeline.models.unet import UNetGenerator
from gan_pipeline.utils.checkpointing import save_checkpoint


def test_load_generator(tmp_path: Path) -> None:
    from gan_pipeline.inference.generate import load_generator

    g = UNetGenerator(in_channels=1, out_channels=3)
    d = UNetGenerator(in_channels=1, out_channels=3)  # dummy discriminator slot
    opt_g = torch.optim.Adam(g.parameters())
    opt_d = torch.optim.Adam(d.parameters())

    ckpt = tmp_path / "gen.pt"
    save_checkpoint(ckpt, 10, g, d, opt_g, opt_d, {"d_loss": 0.1, "g_loss": 0.2})

    g2 = UNetGenerator(in_channels=1, out_channels=3)
    loaded = load_generator(ckpt, g2, torch.device("cpu"))
    assert loaded is g2
    assert not loaded.training  # eval mode


def test_compute_fid_import_error() -> None:
    from gan_pipeline.evaluation.metrics import compute_fid

    with patch.dict("sys.modules", {"torch_fidelity": None}):
        result = compute_fid(Path("/fake/real"), Path("/fake/fake"), torch.device("cpu"))
    import math

    assert math.isnan(result)


def test_compute_inception_score_import_error() -> None:
    from gan_pipeline.evaluation.metrics import compute_inception_score

    with patch.dict("sys.modules", {"torch_fidelity": None}):
        mean, std = compute_inception_score(Path("/fake/fake"), torch.device("cpu"))
    import math

    assert math.isnan(mean) and math.isnan(std)


def test_compute_fid_success() -> None:
    from gan_pipeline.evaluation.metrics import compute_fid

    mock_tf = MagicMock()
    mock_tf.calculate_metrics.return_value = {"frechet_inception_distance": 12.5}

    with patch.dict("sys.modules", {"torch_fidelity": mock_tf}):
        result = compute_fid(Path("/fake/real"), Path("/fake/fake"), torch.device("cpu"))
    assert result == 12.5


def test_compute_inception_score_success() -> None:
    from gan_pipeline.evaluation.metrics import compute_inception_score

    mock_tf = MagicMock()
    mock_tf.calculate_metrics.return_value = {
        "inception_score_mean": 3.2,
        "inception_score_std": 0.4,
    }

    with patch.dict("sys.modules", {"torch_fidelity": mock_tf}):
        mean, std = compute_inception_score(Path("/fake/fake"), torch.device("cpu"))
    assert mean == 3.2 and std == 0.4
