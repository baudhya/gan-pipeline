from pathlib import Path

import torch

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
    import omegaconf

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
