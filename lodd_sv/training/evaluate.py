
from __future__ import annotations

import time
from typing import Optional

import torch

from lodd_sv.engine.forward_process import ForwardProcess
from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
from lodd_sv.math.state_space import DiscreteStateSpace
from lodd_sv.local_coding.local_loss import compute_local_losses
from lodd_sv.local_coding.async_optimizers import LayerWiseOptimizers


def _get_allocated_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0.0


def run_vram_profile(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    batch_size: int = 4,
    seq_len: int = 32,
    num_steps: int = 5,
    use_local_coding: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    device = device or next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    vocab_size = model.vocab_size
    layer_optimizers = None
    if use_local_coding:
        layer_optimizers = LayerWiseOptimizers(model, lr=1e-4)
    alloc_before = _get_allocated_mb(device)
    for _ in range(num_steps):
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        x_t, t = forward_process.corrupt(x_0)
        if use_local_coding and layer_optimizers is not None:
            losses = compute_local_losses(model, x_t, t)
            layer_optimizers.step(losses)
        else:
            logits, _ = model(x_t, t)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), x_0.view(-1), ignore_index=-100
            )
            loss.backward()
            if layer_optimizers is None:
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                opt.step()
                opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alloc_after = _get_allocated_mb(device)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    return {
        "allocated_mb_before": alloc_before,
        "allocated_mb_after": alloc_after,
        "peak_mb": peak_mb,
        "use_local_coding": use_local_coding,
    }


def training_time_per_step(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    batch_size: int,
    seq_len: int,
    use_local_coding: bool,
    device: Optional[torch.device] = None,
    warmup: int = 2,
    steps: int = 10,
) -> float:
    device = device or next(model.parameters()).device
    vocab_size = model.vocab_size
    layer_optimizers = LayerWiseOptimizers(model, lr=1e-4) if use_local_coding else None
    for _ in range(warmup):
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        x_t, t = forward_process.corrupt(x_0)
        if use_local_coding and layer_optimizers is not None:
            losses = compute_local_losses(model, x_t, t)
            layer_optimizers.step(losses)
        else:
            logits, _ = model(x_t, t)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))
            loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        x_t, t = forward_process.corrupt(x_0)
        if use_local_coding and layer_optimizers is not None:
            losses = compute_local_losses(model, x_t, t)
            layer_optimizers.step(losses)
        else:
            logits, _ = model(x_t, t)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))
            loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed / steps


def evaluate_hallucination_rate(
    model: ReverseDenoisingNetwork,
    forward_process: ForwardProcess,
    references: torch.Tensor,
    num_steps: int = 50,
    device: Optional[torch.device] = None,
) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    B, S = references.shape
    vocab_size = model.vocab_size
    with torch.no_grad():
        x_t = torch.randint(0, vocab_size, (B, S), device=device)
        for step in range(num_steps, 0, -1):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            logits, _ = model(x_t, t)
            x_0_pred = logits.argmax(dim=-1)
            x_t = x_0_pred
    match = (x_t == references).float()
    token_acc = match.mean().item()
    return 1.0 - token_acc
