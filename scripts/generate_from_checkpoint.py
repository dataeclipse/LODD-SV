
from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "data" / "lodd_sv_checkpoint.pt"))
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = argmax; >0 = sample (e.g. 0.8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128, help="Must match checkpoint")
    p.add_argument("--num-layers", type=int, default=2, help="Must match checkpoint")
    p.add_argument("--num-heads", type=int, default=2, help="Must match checkpoint")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print("No checkpoint found. Run first: python scripts/train_on_data.py")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]
    vocab_size = len(word2id)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork

    state_space = VocabularyStateSpace(
        vocab_size=vocab_size,
        token_to_id=word2id,
        id_to_token=id2word,
    )
    num_steps = 50
    forward_process = ForwardProcess(state_space, num_steps=num_steps, device=device)
    model = ReverseDenoisingNetwork(
        state_space,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_steps=num_steps,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()


    seq_len = 32
    x_t = torch.randint(0, vocab_size, (args.num_samples, seq_len), device=device)

    with torch.no_grad():
        for step in range(args.steps, 0, -1):
            t = torch.full((args.num_samples,), step, device=device, dtype=torch.long)
            logits, _ = model(x_t, t)
            if args.temperature <= 0:
                x_t = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                x_t = torch.multinomial(probs.view(-1, vocab_size), 1).view(args.num_samples, seq_len)

    id2word_get = id2word.get
    for i in range(args.num_samples):
        tokens = x_t[i].cpu().tolist()
        words = [id2word_get(j, "<unk>") for j in tokens if j != 0]
        text = " ".join(w for w in words if w != "<pad>").strip()
        print(f"[{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}")
    print("Done.")


if __name__ == "__main__":
    main()
