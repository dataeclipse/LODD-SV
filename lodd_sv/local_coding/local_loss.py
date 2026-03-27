
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class LocalPredictiveLoss:

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = pred - target
        if mask is not None:
            diff = diff * mask.unsqueeze(-1)
        loss = (diff ** 2).sum(dim=-1)
        if mask is not None:
            loss = loss * mask
            n = mask.sum().clamp(min=1)
            return loss.sum() / n
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_local_losses(
    model: nn.Module,
    token_ids: torch.Tensor,
    t: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    criterion = LocalPredictiveLoss(reduction="mean")
    stack = model.denoising_stack
    h = stack.token_embed(token_ids)
    t_emb = stack.time_proj(stack.time_embed(t))
    losses = []
    h_prev = h
    for layer in stack.layers:
        h_target = layer(h_prev, t_emb=t_emb, mask=mask)
        h_target_detached = h_target.detach()
        h_pred = layer(h_prev, t_emb=t_emb, mask=mask)
        loss_l = criterion(h_pred, h_target_detached, mask=mask)
        losses.append(loss_l)
        h_prev = h_target.detach()
    return losses
