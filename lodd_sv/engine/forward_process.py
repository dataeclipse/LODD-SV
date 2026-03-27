
from __future__ import annotations

from typing import Optional

import torch

from lodd_sv.math.state_space import DiscreteStateSpace
from lodd_sv.math.diffusion_equations import TransitionMatrixBuilder, get_qt_schedule


class ForwardProcess:

    def __init__(
        self,
        state_space: DiscreteStateSpace,
        num_steps: int,
        beta_max: float = 0.1,
        schedule: str = "linear",
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_space = state_space
        self.num_steps = num_steps
        self.device = device or torch.device("cpu")
        self.betas = get_qt_schedule(
            num_steps, beta_max=beta_max, schedule=schedule, device=self.device
        )





        self._alpha_bar = torch.ones(self.num_steps + 1, device=self.device, dtype=self.betas.dtype)
        for t in range(1, self.num_steps + 1):
            self._alpha_bar[t] = self._alpha_bar[t - 1] * (1.0 - self.betas[t - 1])

    def corrupt(
        self,
        x_0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, S = x_0.shape
        device = x_0.device
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (B,), device=device)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t = t.clamp(1, self.num_steps)
        K = self.state_space.state_dim

        alpha_t = self._alpha_bar[t.long()].to(device=device, dtype=torch.float32)

        keep = (torch.rand((B, S), device=device) < alpha_t.view(B, 1))
        rand_tokens = torch.randint(0, K, (B, S), device=device, dtype=x_0.dtype)
        x_t = torch.where(keep, x_0, rand_tokens)
        return x_t, t

    def posterior_mean_coef(
        self, t: torch.Tensor
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.clamp(0, self.num_steps).long()
        return self._alpha_bar[t]
