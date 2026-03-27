
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def tokenize(text: str, word2id: dict, max_len: int) -> list[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    toks = [word2id.get(w, 1) for w in words[:max_len]]
    return toks + [0] * (max_len - len(toks))


def detokenize(token_ids: list[int], id2word: dict) -> str:
    words = [id2word.get(i, "<unk>") for i in token_ids if i != 0]
    words = [w for w in words if w != "<pad>"]
    return " ".join(words).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DATA_DIR / "lodd_sv_checkpoint.pt"))
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--t-corrupt", type=int, default=10, help="Forward corruption step t (1..T)")
    p.add_argument("--steps", type=int, default=50, help="Reverse denoising steps (<=T recommended)")
    p.add_argument("--temperature", type=float, default=0.7, help="0 = argmax; >0 = sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128, help="Must match checkpoint")
    p.add_argument("--num-layers", type=int, default=2, help="Must match checkpoint")
    p.add_argument("--num-heads", type=int, default=2, help="Must match checkpoint")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"Missing checkpoint: {ckpt_path}")
        raise SystemExit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]
    vocab_size = len(word2id)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork

    state_space = VocabularyStateSpace(vocab_size=vocab_size, token_to_id=word2id, id_to_token=id2word)
    T = 50
    model = ReverseDenoisingNetwork(
        state_space,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_steps=T,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    forward = ForwardProcess(state_space, num_steps=T, device=device)


    x0_ids = tokenize(args.prompt, word2id, args.seq_len)
    x0 = torch.tensor([x0_ids], dtype=torch.long, device=device)
    t_corrupt = max(1, min(int(args.t_corrupt), T))
    t_val = torch.full((1,), t_corrupt, device=device, dtype=torch.long)
    x_t, _ = forward.corrupt(x0, t_val)

    def sample_x_prev_from_posterior(
        *,
        x_t_cur: torch.Tensor,
        x0_hat: torch.Tensor,
        t_int: int,
    ) -> torch.Tensor:


        return x0_hat



    steps = max(1, int(args.steps))
    steps = min(steps, t_corrupt)
    with torch.no_grad():
        for step in range(t_corrupt, t_corrupt - steps, -1):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            logits, _ = model(x_t, t)

            if args.temperature <= 0:
                x0_hat = logits.argmax(dim=-1)
            else:
                probs0 = torch.softmax(logits / args.temperature, dim=-1)
                x0_hat = torch.multinomial(probs0.view(-1, vocab_size), 1).view(1, args.seq_len)

            x_t = sample_x_prev_from_posterior(x_t_cur=x_t, x0_hat=x0_hat, t_int=step)

    out_ids = x_t[0].cpu().tolist()
    print("Prompt :", args.prompt)
    print("Output :", detokenize(out_ids, id2word))
    print("Done.")


if __name__ == "__main__":
    main()

