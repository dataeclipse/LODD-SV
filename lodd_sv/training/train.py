
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn

from lodd_sv.engine.forward_process import ForwardProcess
from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
from lodd_sv.math.state_space import DiscreteStateSpace
from lodd_sv.local_coding.local_loss import compute_local_losses
from lodd_sv.local_coding.async_optimizers import LayerWiseOptimizers


def train_step_global(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    optimizer: torch.optim.Optimizer,
    x_0: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable] = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    x_t, t = forward_process.corrupt(x_0)
    logits, _ = model(x_t, t, mask=mask)
    B, S, V = logits.shape
    loss_fn = loss_fn or nn.functional.cross_entropy
    loss = loss_fn(
        logits.view(-1, V),
        x_0.view(-1),
        ignore_index=0,
        reduction="mean" if mask is None else "none",
    )
    if mask is not None:
        loss = (loss.view(B, S) * mask).sum() / mask.sum().clamp(min=1)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item()}


def train_step_local(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    layer_optimizers: LayerWiseOptimizers,
    x_0: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.train()
    x_t, t = forward_process.corrupt(x_0)
    losses = compute_local_losses(model, x_t, t, mask=mask)
    return layer_optimizers.step(losses)


def train_epoch(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    data_iter: Iterator[torch.Tensor],
    use_local_coding: bool,
    global_optimizer: Optional[torch.optim.Optimizer] = None,
    layer_optimizers: Optional[LayerWiseOptimizers] = None,
    mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    agg: Dict[str, float] = {}
    n = 0
    for batch in data_iter:
        if not isinstance(batch, torch.Tensor):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x_0, mask = batch[0], batch[1]
                x_0 = x_0.to(device)
                mask = mask.to(device) if mask is not None else None
            else:
                x_0 = batch[0] if isinstance(batch, (list, tuple)) else batch["input_ids"]
                x_0 = x_0.to(device)
                mask = mask_fn(x_0) if mask_fn else None
        else:
            x_0 = batch.to(device)
            mask = mask_fn(x_0) if mask_fn else None
        if use_local_coding and layer_optimizers is not None:
            step_metrics = train_step_local(model, forward_process, layer_optimizers, x_0, mask)
        else:
            if global_optimizer is None:
                raise ValueError("global_optimizer required when use_local_coding=False")
            step_metrics = train_step_global(model, forward_process, global_optimizer, x_0, mask)
        for k, v in step_metrics.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        if max_steps is not None and n >= max_steps:
            break
    if n > 0:
        for k in agg:
            agg[k] /= n
    return agg
