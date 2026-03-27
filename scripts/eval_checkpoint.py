
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_csv_texts(path: Path, text_col: str = "text", max_rows: int = 0):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if text_col not in fieldnames:
            lowered = {fn.lower(): fn for fn in fieldnames}
            if text_col.lower() in lowered:
                text_col = lowered[text_col.lower()]
            elif len(fieldnames) == 1:
                text_col = fieldnames[0]

        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            t = (row.get(text_col) or f"{row.get('question','')} {row.get('answer','')}").strip()
            if t:
                rows.append(t)
    return rows


def tokenize(text: str, word2id: dict, max_len: int):
    words = re.findall(r"[a-z0-9]+", text.lower())
    toks = [word2id.get(w, 1) for w in words[:max_len]]
    return toks + [0] * (max_len - len(toks))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DATA_DIR / "lodd_sv_checkpoint.pt"))
    p.add_argument("--data", type=str, default=str(DATA_DIR / "simple_qa_verified.csv"))
    p.add_argument("--text-col", type=str, default="text", help="CSV column to read as text (default: text)")
    p.add_argument("--mode", type=str, default="reconstruct", choices=("reconstruct", "generate"),
                   help="reconstruct: corrupt ref then denoise (measures denoising). generate: from noise (often ~0%%).")
    p.add_argument("--n", type=int, default=50, help="Number of test examples")
    p.add_argument("--steps", type=int, default=40, help="Denoising steps (generate mode)")
    p.add_argument("--t-corrupt", type=int, default=0, help="Reconstruct: corruption step t (default 25). Use 5–10 for easier task.")
    p.add_argument("--debug", action="store_true", help="Print first sample ref vs pred and prediction stats.")
    p.add_argument("--d-model", type=int, default=128, help="Model hidden size (must match checkpoint)")
    p.add_argument("--num-layers", type=int, default=2, help="Number of layers (must match checkpoint)")
    p.add_argument("--num-heads", type=int, default=2, help="Number of heads (must match checkpoint)")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    data_path = Path(args.data)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not ckpt_path.exists() or not data_path.exists():
        print("Missing checkpoint or data CSV.")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]
    vocab_size = len(word2id)

    texts = load_csv_texts(data_path, text_col=args.text_col, max_rows=args.n)
    if not texts:
        print(f"No rows loaded from {data_path.name}. Try --text-col <column_name>.")
        raise SystemExit(1)
    seq_len = 32
    ref_ids = [tokenize(t, word2id, seq_len) for t in texts]
    ref = torch.tensor(ref_ids, dtype=torch.long, device=device)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork

    state_space = VocabularyStateSpace(vocab_size=vocab_size, token_to_id=word2id, id_to_token=id2word)
    num_steps = 50
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
    forward_process = ForwardProcess(state_space, num_steps=num_steps, device=device)

    B = ref.size(0)
    t_corrupt = 0
    if args.mode == "reconstruct":

        t_corrupt = args.t_corrupt if args.t_corrupt > 0 else min(num_steps // 2, 25)
        t_corrupt = max(1, min(t_corrupt, num_steps))
        t_val = torch.full((B,), t_corrupt, device=device, dtype=torch.long)
        x_t, _ = forward_process.corrupt(ref, t_val)
        with torch.no_grad():
            logits, _ = model(x_t, t_val)
            x_t = logits.argmax(dim=-1)
    else:
        x_t = torch.randint(0, vocab_size, (B, seq_len), device=device)
        with torch.no_grad():
            for step in range(args.steps, 0, -1):
                t = torch.full((B,), step, device=device, dtype=torch.long)
                logits, _ = model(x_t, t)
                x_t = logits.argmax(dim=-1)

    match = (x_t == ref).float()

    mask = (ref != 0).float()
    n_valid = mask.sum().item()
    correct = (match * mask).sum().item()
    token_acc = correct / n_valid if n_valid else 0.0
    hallucination_proxy = 1.0 - token_acc

    pred = x_t
    if getattr(args, "debug", False):
        print("[debug] n_valid =", n_valid)
        uniq = pred.unique().cpu().tolist()
        print("[debug] num unique pred tokens =", len(uniq), "sample:", uniq[:20])
        print("[debug] pred mode (most common) =", pred.flatten().mode().values.item())
        r0, p0 = ref[0].cpu().tolist(), pred[0].cpu().tolist()
        id2w = id2word.get
        print("[debug] first sample ref[:12] =", [id2w(i, "?") for i in r0[:12]])
        print("[debug] first sample pred[:12] =", [id2w(i, "?") for i in p0[:12]])
        matches_first = sum(1 for i in range(min(12, seq_len)) if r0[i] != 0 and r0[i] == p0[i])
        print("[debug] matches in first 12 non-pad =", matches_first)

    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Data: {data_path.name} (n={len(texts)})  mode={args.mode}" + (f"  t_corrupt={t_corrupt}" if args.mode == "reconstruct" else ""))
    print(f"Token accuracy: {token_acc:.4f}")
    print(f"Hallucination proxy (1 - acc): {hallucination_proxy:.4f}")
    if token_acc == 0.0 and args.mode == "reconstruct":
        print("(Tip: 0%% is common if the model was trained with local predictive coding only (default).")
        print(" The head was not trained to predict x_0. Train with --no-local to get non-zero reconstruction.)")
    print("Done.")


if __name__ == "__main__":
    main()
