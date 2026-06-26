from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from gan_pipeline.models.multiscale_disc import MultiScaleDiscriminator


class LossType(str, Enum):
    BCE = "bce"
    WASSERSTEIN = "wasserstein"
    HINGE = "hinge"
    LSGAN = "lsgan"


def generator_loss(fake_logits: torch.Tensor, loss_type: LossType) -> torch.Tensor:
    if loss_type == LossType.BCE:
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if loss_type == LossType.WASSERSTEIN:
        return -fake_logits.mean()
    if loss_type == LossType.HINGE:
        return -fake_logits.mean()
    if loss_type == LossType.LSGAN:
        return 0.5 * F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    raise ValueError(f"Unknown loss type: {loss_type}")


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_type: LossType,
    label_smoothing: float = 1.0,
) -> torch.Tensor:
    if loss_type == LossType.BCE:
        real_targets = torch.full_like(real_logits, label_smoothing)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) / 2
    if loss_type == LossType.WASSERSTEIN:
        return fake_logits.mean() - real_logits.mean()
    if loss_type == LossType.HINGE:
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if loss_type == LossType.LSGAN:
        real_targets = torch.full_like(real_logits, label_smoothing)
        return 0.5 * (
            F.mse_loss(real_logits, real_targets)
            + F.mse_loss(fake_logits, torch.zeros_like(fake_logits))
        )
    raise ValueError(f"Unknown loss type: {loss_type}")


def multiscale_discriminator_loss(
    real_list: list[torch.Tensor],
    fake_list: list[torch.Tensor],
    loss_type: LossType,
    label_smoothing: float = 1.0,
) -> torch.Tensor:
    """Average discriminator loss across all scales."""
    losses = [
        discriminator_loss(r, f, loss_type, label_smoothing) for r, f in zip(real_list, fake_list)
    ]
    return torch.stack(losses).mean()


def multiscale_generator_loss(
    fake_list: list[torch.Tensor],
    loss_type: LossType,
) -> torch.Tensor:
    """Average generator adversarial loss across all scales."""
    losses = [generator_loss(f, loss_type) for f in fake_list]
    return torch.stack(losses).mean()


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    """L1 distance between real and fake discriminator features, averaged over scales and layers.

    Args:
        real_features: per-scale list of intermediate feature maps from the real pair.
        fake_features: same structure for the fake pair (gradients flow through these).
    Both arguments come from MultiScaleDiscriminator.forward_with_features().
    """
    total = fake_features[0][0].new_zeros(())
    n = 0
    for real_scale, fake_scale in zip(real_features, fake_features):
        for real_feat, fake_feat in zip(real_scale, fake_scale):
            total = total + F.l1_loss(fake_feat, real_feat.detach())
            n += 1
    return total / max(n, 1)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG16 features (relu1_2, relu2_2, relu3_3, relu4_3).

    Inputs are expected in [-1, 1]; they are rescaled to ImageNet-normalised [0, 1]
    before being passed through VGG.  Arbitrary channel counts are handled: single-channel
    tensors are expanded to 3; tensors with more than 3 channels are truncated to the
    first 3 (RGB).
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, weights_path: str | None = None) -> None:
        super().__init__()
        vgg = torchvision.models.vgg16(weights=None)
        if weights_path is not None:
            from pathlib import Path

            wpath = Path(weights_path)
            if not wpath.exists():
                raise FileNotFoundError(
                    f"VGG weights not found at '{weights_path}'. "
                    "Run `make download-weights` to fetch them into weights/."
                )
            state = torch.load(str(wpath), map_location="cpu", weights_only=True)
            vgg.load_state_dict(state)
        else:
            vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features
        self.slice1 = feats[:4]  # relu1_2
        self.slice2 = feats[4:9]  # relu2_2
        self.slice3 = feats[9:16]  # relu3_3
        self.slice4 = feats[16:23]  # relu4_3
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0  # [-1, 1] → [0, 1]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_p, real_p = self._preprocess(fake), self._preprocess(real)
        loss = fake_p.new_zeros(())
        for slice_ in (self.slice1, self.slice2, self.slice3, self.slice4):
            fake_p = slice_(fake_p)
            real_p = slice_(real_p)
            loss = loss + F.l1_loss(fake_p, real_p)
        return loss


def multiscale_gradient_penalty(
    discriminator: MultiScaleDiscriminator,
    real_pair: torch.Tensor,
    fake_pair: torch.Tensor,
) -> torch.Tensor:
    """WGAN-GP gradient penalty averaged across all discriminator scales.

    Computes per-scale GP using the same interpolated input at each resolution,
    then averages — so each scale's Lipschitz constraint is enforced independently.
    """
    batch = real_pair.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=real_pair.device)
    interp = (alpha * real_pair + (1 - alpha) * fake_pair.detach()).requires_grad_(True)
    logits = discriminator(interp)  # list[Tensor], finest → coarsest

    gp_per_scale: list[torch.Tensor] = []
    for logit in logits:
        grads = torch.autograd.grad(
            outputs=logit.sum(),
            inputs=interp,
            create_graph=True,
            retain_graph=True,  # keep activations alive until d_loss.backward() frees them
        )[0]
        gp_per_scale.append(((grads.view(batch, -1).norm(2, dim=1) - 1) ** 2).mean())

    return torch.stack(gp_per_scale).mean()


def r1_gradient_penalty(
    discriminator: MultiScaleDiscriminator,
    real_pair: torch.Tensor,
) -> torch.Tensor:
    """R1 regularization: penalizes the squared gradient norm at real samples.

    Prevents D from becoming arbitrarily sharp near real data without constraining
    it on fakes (unlike WGAN-GP). Effective for BCE, LSGAN, and hinge losses.
    """
    real_rg = real_pair.detach().requires_grad_(True)
    real_maps = discriminator(real_rg)

    gp_per_scale: list[torch.Tensor] = []
    for logit in real_maps:
        grads = torch.autograd.grad(
            outputs=logit.sum(),
            inputs=real_rg,
            create_graph=True,
            retain_graph=True,
        )[0]
        gp_per_scale.append(grads.pow(2).view(grads.size(0), -1).sum(dim=1).mean())

    return torch.stack(gp_per_scale).mean()
