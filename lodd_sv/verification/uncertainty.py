
from __future__ import annotations

from typing import Optional

import torch


def entropy_per_position(probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    log_p = torch.log(probs.clamp(min=1e-10))
    H = -(probs * log_p).sum(dim=-1)
    if mask is not None:
        H = H * mask
    return H


def variance_per_position(probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    var = 1.0 - (probs ** 2).sum(dim=-1)
    if mask is not None:
        var = var * mask
    return var


def uncertainty_score(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    method: str = "entropy",
    aggregate: str = "max",
) -> torch.Tensor:
    if method == "entropy":
        U = entropy_per_position(probs, mask=mask)
    else:
        U = variance_per_position(probs, mask=mask)
    if aggregate == "max":
        u = U.max(dim=1).values
    elif aggregate == "mean":
        if mask is not None:
            u = (U.sum(dim=1) / mask.sum(dim=1).clamp(min=1))
        else:
            u = U.mean(dim=1)
    else:
        u = U.sum(dim=1)
    return u
