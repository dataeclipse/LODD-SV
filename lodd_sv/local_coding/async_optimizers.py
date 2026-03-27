
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn


class LayerWiseOptimizers:

    def __init__(
        self,
        model: nn.Module,
        layer_params: Optional[List[List[nn.Parameter]]] = None,
        lr: float = 1e-4,
        optimizer_class: type = torch.optim.AdamW,
        **optimizer_kwargs: Any,
    ) -> None:
        self.model = model
        if layer_params is None:
            layer_params = self._default_layer_params()
        self.optimizers: List[torch.optim.Optimizer] = []
        for params in layer_params:
            opt = optimizer_class(params, lr=lr, **optimizer_kwargs)
            self.optimizers.append(opt)

    def _default_layer_params(self) -> List[List[nn.Parameter]]:
        stack = getattr(self.model, "denoising_stack", self.model)
        if hasattr(stack, "layers"):
            return [list(layer.parameters()) for layer in stack.layers]
        return [list(self.model.parameters())]

    def zero_grad_all(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad()

    def step_layer(self, layer_idx: int, loss: torch.Tensor) -> None:
        self.optimizers[layer_idx].zero_grad()
        loss.backward(retain_graph=(layer_idx < len(self.optimizers) - 1))
        self.optimizers[layer_idx].step()

    def step(self, losses: List[torch.Tensor]) -> Dict[str, float]:
        self.zero_grad_all()
        out = {}
        for l, loss in enumerate(losses):
            if loss.requires_grad:
                loss.backward(retain_graph=(l < len(losses) - 1))
            out[f"loss_l{l}"] = loss.detach().item()
        for opt in self.optimizers:
            opt.step()
        return out
