from pathlib import Path
from unittest.mock import MagicMock, patch

import omegaconf
import torch
from torch.utils.data import DataLoader, TensorDataset

from gan_pipeline.models import DCGANDiscriminator, DCGANGenerator
from gan_pipeline.training.trainer import GANTrainer
from gan_pipeline.utils.checkpointing import load_checkpoint, save_checkpoint


def test_save_load_checkpoint(tmp_path: Path) -> None:
    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    opt_g = torch.optim.Adam(g.parameters())
    opt_d = torch.optim.Adam(d.parameters())

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        epoch=5,
        generator=g,
        discriminator=d,
        opt_g=opt_g,
        opt_d=opt_d,
        metrics={"d_loss": 0.5, "g_loss": 0.6},
    )

    state = load_checkpoint(ckpt, torch.device("cpu"))
    assert state["epoch"] == 5
    assert "generator" in state and "discriminator" in state
    assert state["metrics"] == {"d_loss": 0.5, "g_loss": 0.6}


def test_trainer_step(cfg, device: torch.device, tmp_path: Path) -> None:
    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)

    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    trainer = GANTrainer(g, d, cfg, device, tmp_path)

    real = torch.randn(4, 3, 64, 64)
    d_loss, g_loss = trainer._train_step(real)

    assert isinstance(d_loss, float) and isinstance(g_loss, float)
    assert not (d_loss != d_loss)  # not NaN
    assert not (g_loss != g_loss)


def test_gan_trainer_gradient_penalty(cfg, device: torch.device, tmp_path: Path) -> None:
    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)
        cfg.training.gradient_penalty = True

    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    trainer = GANTrainer(g, d, cfg, device, tmp_path)

    real = torch.randn(4, 3, 64, 64)
    d_loss, g_loss = trainer._train_step(real)
    assert isinstance(d_loss, float) and not (d_loss != d_loss)


def test_gan_trainer_resume(cfg, device: torch.device, tmp_path: Path) -> None:
    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)

    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    trainer = GANTrainer(g, d, cfg, device, tmp_path)

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, 3, g, d, trainer.opt_g, trainer.opt_d, {"d_loss": 0.1, "g_loss": 0.2})
    trainer.resume(ckpt)
    assert trainer.start_epoch == 4


def test_gan_trainer_save_samples(cfg, device: torch.device, tmp_path: Path) -> None:
    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)

    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    trainer = GANTrainer(g, d, cfg, device, tmp_path)
    trainer._save_samples(0)
    assert (tmp_path / "samples" / "epoch_0000.png").exists()


def test_gan_trainer_train_loop(cfg, device: torch.device, tmp_path: Path) -> None:
    with omegaconf.open_dict(cfg):
        cfg.output_dir = str(tmp_path)
        cfg.training.epochs = 1

    g = DCGANGenerator(latent_dim=100, channels=3, image_size=64)
    d = DCGANDiscriminator(channels=3, image_size=64)
    trainer = GANTrainer(g, d, cfg, device, tmp_path)

    dataset = TensorDataset(torch.randn(4, 3, 64, 64), torch.zeros(4, dtype=torch.long))
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset, batch_size=4)

    with patch("gan_pipeline.training.trainer.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value = MagicMock()
        trainer.train(loader)

    assert (tmp_path / "samples" / "epoch_0000.png").exists()
    assert (tmp_path / "checkpoints" / "epoch_0000.pt").exists()
