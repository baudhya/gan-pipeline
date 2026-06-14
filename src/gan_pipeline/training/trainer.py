from pathlib import Path

import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan_pipeline.models.base import BaseDiscriminator, BaseGenerator
from gan_pipeline.models.losses import (
    LossType,
    discriminator_loss,
    generator_loss,
    gradient_penalty,
)
from gan_pipeline.utils.checkpointing import load_checkpoint, save_checkpoint


class GANTrainer:
    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: BaseDiscriminator,
        cfg: DictConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.loss_type = LossType(cfg.training.loss_type)

        self.opt_g = torch.optim.Adam(
            generator.parameters(),
            lr=cfg.training.lr_generator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )
        self.opt_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=cfg.training.lr_discriminator,
            betas=(cfg.training.beta1, cfg.training.beta2),
        )

        self.fixed_z = torch.randn(64, cfg.model.latent_dim, device=device)
        self.start_epoch = 0

        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def resume(self, checkpoint_path: Path) -> None:
        state = load_checkpoint(checkpoint_path, self.device)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.start_epoch = state["epoch"] + 1
        logger.info(f"Resumed from epoch {state['epoch']}")

    def _train_step(self, real: torch.Tensor) -> tuple[float, float]:
        real = real.to(self.device)
        batch_size = real.size(0)

        # Discriminator update (n_critic times)
        for _ in range(self.cfg.training.n_critic):
            z = torch.randn(batch_size, self.cfg.model.latent_dim, device=self.device)
            fake = self.generator(z).detach()

            real_logits = self.discriminator(real)
            fake_logits = self.discriminator(fake)
            d_loss = discriminator_loss(real_logits, fake_logits, self.loss_type)

            if self.cfg.training.gradient_penalty:
                gp = gradient_penalty(self.discriminator, real, fake, self.device)
                d_loss = d_loss + self.cfg.training.gradient_penalty_lambda * gp

            self.opt_d.zero_grad()
            d_loss.backward()  # type: ignore[no-untyped-call]
            self.opt_d.step()

        # Generator update
        z = torch.randn(batch_size, self.cfg.model.latent_dim, device=self.device)
        fake = self.generator(z)
        fake_logits = self.discriminator(fake)
        g_loss = generator_loss(fake_logits, self.loss_type)

        self.opt_g.zero_grad()
        g_loss.backward()  # type: ignore[no-untyped-call]
        self.opt_g.step()

        return d_loss.item(), g_loss.item()

    def _save_samples(self, epoch: int) -> None:
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(self.fixed_z)
        self.generator.train()
        samples = (samples + 1) / 2  # [-1,1] -> [0,1]
        save_image(samples, self.output_dir / "samples" / f"epoch_{epoch:04d}.png", nrow=8)

    def train(self, dataloader: DataLoader) -> None:  # type: ignore[type-arg]
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.cfg.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "model": self.cfg.model.name,
                    "latent_dim": self.cfg.model.latent_dim,
                    "loss_type": self.cfg.training.loss_type,
                    "lr_g": self.cfg.training.lr_generator,
                    "lr_d": self.cfg.training.lr_discriminator,
                    "batch_size": self.cfg.training.batch_size,
                }
            )

            for epoch in range(self.start_epoch, self.cfg.training.epochs):
                self.generator.train()
                self.discriminator.train()

                d_losses: list[float] = []
                g_losses: list[float] = []

                for i, (real, _) in enumerate(dataloader):
                    d_loss, g_loss = self._train_step(real)
                    d_losses.append(d_loss)
                    g_losses.append(g_loss)

                    if i % self.cfg.training.log_every == 0:
                        logger.info(
                            f"Epoch {epoch}/{self.cfg.training.epochs} "
                            f"[{i}/{len(dataloader)}] "
                            f"D: {d_loss:.4f}  G: {g_loss:.4f}"
                        )

                avg_d = sum(d_losses) / len(d_losses)
                avg_g = sum(g_losses) / len(g_losses)
                mlflow.log_metrics({"d_loss": avg_d, "g_loss": avg_g}, step=epoch)

                if epoch % self.cfg.training.sample_every == 0:
                    self._save_samples(epoch)

                if epoch % self.cfg.training.save_every == 0:
                    save_checkpoint(
                        self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                        epoch,
                        self.generator,
                        self.discriminator,
                        self.opt_g,
                        self.opt_d,
                        {"d_loss": avg_d, "g_loss": avg_g},
                    )
