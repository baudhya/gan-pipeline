from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseGenerator(nn.Module, ABC):
    latent_dim: int

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor: ...

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            return self(z)


class BaseDiscriminator(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
