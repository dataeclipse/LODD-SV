
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from lodd_sv.math.state_space import VocabularyStateSpace
from lodd_sv.engine.forward_process import ForwardProcess
from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
from lodd_sv.local_coding.async_optimizers import LayerWiseOptimizers
from lodd_sv.training.train import train_step_local, train_step_global
from lodd_sv.verification import InMemoryKnowledgeBase, StatisticalRouter


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_size = 512
    batch_size = 4
    seq_len = 32
    num_steps = 50

    state_space = VocabularyStateSpace(vocab_size=vocab_size)
    forward_process = ForwardProcess(
        state_space, num_steps=num_steps, device=device
    )
    model = ReverseDenoisingNetwork(
        state_space,
        d_model=128,
        num_layers=2,
        num_heads=2,
        num_steps=num_steps,
    ).to(device)

    layer_optimizers = LayerWiseOptimizers(model, lr=1e-4)
    x_0 = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    metrics_local = train_step_local(model, forward_process, layer_optimizers, x_0)
    print("Local coding step metrics:", metrics_local)

    model2 = ReverseDenoisingNetwork(
        state_space, d_model=128, num_layers=2, num_heads=2, num_steps=num_steps
    ).to(device)
    global_opt = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    metrics_global = train_step_global(model2, forward_process, global_opt, x_0)
    print("Global loss step metrics:", metrics_global)

    kb = InMemoryKnowledgeBase()
    kb.add("demo_fact", "The quick brown fox jumps over the lazy dog.")
    router = StatisticalRouter(kb, threshold=2.0)
    probs = model.predict_probs(x_0[:1], torch.full((1,), num_steps // 2, device=device))
    result = router(probs, x_0[:1])
    print("Router result:", result.triggered, result.uncertainty_value, result.retrieved_fact)

    if device.type == "cuda":
        from lodd_sv.training.evaluate import run_vram_profile
        profile = run_vram_profile(
            model, forward_process,
            batch_size=batch_size, seq_len=seq_len,
            num_steps=3, use_local_coding=True, device=device,
        )
        print("VRAM (local coding):", profile)

    print("Demo finished.")


if __name__ == "__main__":
    main()
