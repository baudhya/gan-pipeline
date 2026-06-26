from abc import ABC, abstractmethod
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from gan_pipeline.models.base import BaseGenerator
from gan_pipeline.utils.checkpointing import load_checkpoint, save_checkpoint


class BaseTrainer(ABC):
    """Abstract base for GAN trainers.

    Subclasses must implement:
        _log_params()   — hyperparameters to record in MLflow at run start
        _step_batch()   — one forward/backward pass; returns a metric dict
        _save_samples() — write visualisation images for an epoch

    Optional hooks (no-ops by default):
        _build_schedulers()       — called once after optimizers are created
        _step_schedulers()        — called at the end of every epoch
        _restore_schedulers()     — called after resume() to rewind schedulers
    """

    def __init__(
        self,
        generator: BaseGenerator,
        discriminator: nn.Module,
        cfg: DictConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.start_epoch = 0

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

        self._build_schedulers()

        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Scheduler hooks
    # ------------------------------------------------------------------

    def _build_schedulers(self) -> None:
        """Create LR schedulers. Called once after optimizers exist."""

    def _step_schedulers(self) -> None:
        """Step schedulers. Called at the end of every epoch."""

    def _restore_schedulers(self, start_epoch: int) -> None:
        """Restore scheduler state after loading a checkpoint."""

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def resume(self, checkpoint_path: Path) -> None:
        state = load_checkpoint(checkpoint_path, self.device)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.start_epoch = state["epoch"] + 1
        self._restore_schedulers(self.start_epoch)
        logger.info(f"Resumed from epoch {state['epoch']}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _log_params(self) -> dict[str, object]:
        """Return hyperparameters to log to MLflow at the start of training."""

    @abstractmethod
    def _step_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one forward/backward pass. Return a flat dict of scalar metrics."""

    @abstractmethod
    def _save_samples(self, epoch: int) -> None:
        """Save visualisation images for this epoch."""

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader: DataLoader) -> None:  # type: ignore[type-arg]
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.cfg.experiment_name)

        with mlflow.start_run():
            mlflow.log_params(self._log_params())

            for epoch in range(self.start_epoch, self.cfg.training.epochs):
                self.generator.train()
                self.discriminator.train()

                accum: dict[str, list[float]] = {}

                for i, batch in enumerate(dataloader):
                    metrics = self._step_batch(batch)

                    for k, v in metrics.items():
                        accum.setdefault(k, []).append(v)

                    if i % self.cfg.training.log_every == 0:
                        parts = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                        logger.info(
                            f"Epoch {epoch}/{self.cfg.training.epochs} "
                            f"[{i}/{len(dataloader)}]  {parts}"
                        )
                        self._save_samples(epoch * 10000 + i)

                avgs = {k: sum(vs) / len(vs) for k, vs in accum.items()}
                mlflow.log_metrics(avgs, step=epoch)

                if epoch % self.cfg.training.sample_every == 0:
                    self._save_samples(epoch)
                    mlflow.log_artifact(
                        str(self.output_dir / "samples" / f"epoch_{epoch:04d}.png"),
                        artifact_path="samples",
                    )

                if epoch % self.cfg.training.save_every == 0:
                    save_checkpoint(
                        self.output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                        epoch,
                        self.generator,
                        self.discriminator,
                        self.opt_g,
                        self.opt_d,
                        avgs,
                    )

                self._step_schedulers()
