import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "model": {
                "name": "pix2pix",
                "generator": {"base_features": 64},
                "discriminator": {"base_features": 64, "n_scales": 1},
            },
            "training": {
                "loss_type": "hinge",
                "lr_generator": 0.0002,
                "lr_discriminator": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999,
                "lambda_l1": 100.0,
                "lambda_vgg": 0.0,
                "lambda_fm": 0.0,
                "lambda_gp": 0.0,
                "label_smoothing": 1.0,
                "batch_size": 1,
                "epochs": 2,
                "log_every": 1,
                "sample_every": 1,
                "save_every": 1,
                "num_workers": 0,
            },
            "data": {
                "image_size": 256,
                "sar_channels": 1,
                "eo_channels": 3,
            },
            "experiment_name": "test",
            "output_dir": "test_outputs",
        }
    )
