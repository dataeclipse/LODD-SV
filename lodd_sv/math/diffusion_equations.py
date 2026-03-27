
from __future__ import annotations

from typing import Callable, Optional

import torch


class TransitionMatrixBuilder:

    def __init__(self, num_states: int) -> None:
        self.K = num_states

    def uniform_transition(self, beta_t: torch.Tensor) -> torch.Tensor:
        device = beta_t.device
        if beta_t.dim() == 0:
            beta_t = beta_t.unsqueeze(0)
        b = beta_t.view(-1, 1, 1)
        I = torch.eye(self.K, device=device, dtype=beta_t.dtype)
        ones = torch.ones(self.K, self.K, device=device, dtype=beta_t.dtype) / self.K
        Q = (1 - b) * I + b * ones
        return Q

    def product_transition(self, betas: torch.Tensor) -> torch.Tensor:
        Q_one = self.uniform_transition(betas[0])
        out = Q_one.clone()
        for i in range(1, betas.shape[0]):
            Qt = self.uniform_transition(betas[i])
            out = torch.einsum("kij,kjl->kil", out.unsqueeze(0), Qt.unsqueeze(0)).squeeze(0)
        return out


def continuous_time_limit_beta(
    t: torch.Tensor,
    beta_max: float = 0.1,
    schedule: str = "linear",
) -> torch.Tensor:
    if schedule == "linear":
        return beta_max * t
    if schedule == "cosine":
        s = 0.008
        return beta_max * torch.cos(((t + s) / (1 + s)) * (3.14159 / 2)) ** 2
    if schedule == "sqrt":
        return beta_max * torch.sqrt(t)
    return beta_max * t


def get_qt_schedule(
    num_steps: int,
    beta_max: float = 0.1,
    schedule: str = "linear",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    t = torch.linspace(1.0 / num_steps, 1.0, num_steps, device=device)
    beta = continuous_time_limit_beta(t, beta_max=beta_max, schedule=schedule)
    return beta
