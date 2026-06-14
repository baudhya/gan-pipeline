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
                "name": "dcgan",
                "latent_dim": 100,
                "generator": {"base_features": 64},
                "discriminator": {"base_features": 64},
            },
            "training": {
                "loss_type": "bce",
                "lr_generator": 0.0002,
                "lr_discriminator": 0.0002,
                "beta1": 0.5,
                "beta2": 0.999,
                "n_critic": 1,
                "gradient_penalty": False,
                "gradient_penalty_lambda": 10,
                "batch_size": 4,
                "epochs": 2,
                "log_every": 1,
                "sample_every": 1,
                "save_every": 1,
                "num_workers": 0,
            },
            "data": {
                "image_size": 64,
                "channels": 3,
            },
            "experiment_name": "test",
            "output_dir": "test_outputs",
        }
    )
