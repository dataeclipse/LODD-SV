
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from lodd_sv.math.layer_blocks import DenoisingStack
from lodd_sv.math.state_space import DiscreteStateSpace


class ReverseDenoisingNetwork(nn.Module):

    def __init__(
        self,
        state_space: DiscreteStateSpace,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        num_steps: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_space = state_space
        self.vocab_size = state_space.state_dim
        self.num_steps = num_steps
        self.denoising_stack = DenoisingStack(
            vocab_size=self.vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            max_steps=num_steps,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hiddens: bool = False,
    ) -> tuple[torch.Tensor, Optional[list]]:
        logits, hiddens = self.denoising_stack(x_t, t, mask=mask, return_hiddens=return_hiddens)
        return logits, hiddens

    def predict_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x_t, t)
        return logits

    def predict_probs(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(x_t, t)
        return torch.softmax(logits, dim=-1)
