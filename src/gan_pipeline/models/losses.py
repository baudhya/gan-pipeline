import math
from enum import Enum

import torch
import torch.nn.functional as F

from gan_pipeline.models.base import BaseDiscriminator


class LossType(str, Enum):
    BCE = "bce"
    WASSERSTEIN = "wasserstein"
    HINGE = "hinge"


def generator_loss(fake_logits: torch.Tensor, loss_type: LossType) -> torch.Tensor:
    if loss_type == LossType.BCE:
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if loss_type == LossType.WASSERSTEIN:
        return -fake_logits.mean()
    if loss_type == LossType.HINGE:
        return -fake_logits.mean()
    raise ValueError(f"Unknown loss type: {loss_type}")


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_type: LossType,
) -> torch.Tensor:
    if loss_type == LossType.BCE:
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) / 2
    if loss_type == LossType.WASSERSTEIN:
        return fake_logits.mean() - real_logits.mean()
    if loss_type == LossType.HINGE:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    raise ValueError(f"Unknown loss type: {loss_type}")


def multiscale_discriminator_loss(
    real_list: list[torch.Tensor],
    fake_list: list[torch.Tensor],
    loss_type: LossType,
) -> torch.Tensor:
    """Average discriminator loss across all scales."""
    losses = [discriminator_loss(r, f, loss_type) for r, f in zip(real_list, fake_list)]
    return torch.stack(losses).mean()


def multiscale_generator_loss(
    fake_list: list[torch.Tensor],
    loss_type: LossType,
) -> torch.Tensor:
    """Average generator adversarial loss across all scales."""
    losses = [generator_loss(f, loss_type) for f in fake_list]
    return torch.stack(losses).mean()


def gradient_penalty(
    discriminator: BaseDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

    logits = discriminator(interpolated)
    grad = torch.autograd.grad(
        outputs=logits,
        inputs=interpolated,
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(batch_size, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()
