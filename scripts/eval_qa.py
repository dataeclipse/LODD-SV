
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_qa_pairs(path: Path, question_col: str = "question", answer_col: str = "answer", text_col: str = "text", max_rows: int = 0):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = {name.lower(): name for name in (r.fieldnames or [])}
        q_key = fieldnames.get(question_col.lower(), question_col)
        a_key = fieldnames.get(answer_col.lower(), answer_col)
        t_key = fieldnames.get(text_col.lower(), text_col)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            q = (row.get(q_key) or "").strip()
            a = (row.get(a_key) or "").strip()
            if not q or not a:
                t = (row.get(t_key) or "").strip()
                if t and "?" in t:
                    idx = t.index("?") + 1
                    q, a = t[:idx].strip(), t[idx:].strip()
                elif t:

                    q = q or ""
                    a = a or t
            if a:
                pairs.append((q, a))
    return pairs


def tokenize_words(text: str, word2id: dict, max_len: int) -> list[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_len]
    return [word2id.get(w, 1) for w in words] + [0] * (max_len - len(words))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DATA_DIR / "lodd_sv_checkpoint.pt"))
    p.add_argument("--data", type=str, default=str(DATA_DIR / "simple_qa_verified.csv"))
    p.add_argument("--t-corrupt", type=int, default=5)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=32)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not ckpt_path.exists() or not data_path.exists():
        print("Missing checkpoint or data.")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]
    vocab_size = len(word2id)
    max_q_len = getattr(args, "max_q_len", None) or (args.seq_len // 2)
    max_a_len = (args.seq_len - max_q_len)

    pairs = load_qa_pairs(data_path, max_rows=args.n)
    if not pairs:
        print("No QA pairs loaded.")
        raise SystemExit(2)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork

    state_space = VocabularyStateSpace(vocab_size=vocab_size, token_to_id=word2id, id_to_token=id2word)
    num_steps = 50
    forward = ForwardProcess(state_space, num_steps=num_steps, device=device)
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

    t_corrupt = max(1, min(args.t_corrupt, num_steps))
    refs = []
    answer_masks = []
    for q, a in pairs:
        q_toks = tokenize_words(q, word2id, max_q_len)
        a_toks = tokenize_words(a, word2id, max_a_len)
        seq = q_toks + a_toks
        refs.append(seq)
        mask = [0.0] * max_q_len + [1.0 if a_toks[i] != 0 else 0.0 for i in range(max_a_len)]
        answer_masks.append(mask)

    ref = torch.tensor(refs, dtype=torch.long, device=device)
    t_val = torch.full((ref.size(0),), t_corrupt, device=device, dtype=torch.long)
    x_t, _ = forward.corrupt(ref, t_val)
    with torch.no_grad():
        logits, _ = model(x_t, t_val)
        pred = logits.argmax(dim=-1)

    match = (pred == ref).float()
    full_mask = (ref != 0).float()
    answer_mask_t = torch.tensor(answer_masks, device=device, dtype=torch.float)
    n_full = full_mask.sum().item()
    n_answer = answer_mask_t.sum().item()
    correct_full = (match * full_mask).sum().item()
    correct_answer = (match * answer_mask_t).sum().item()
    full_acc = correct_full / n_full if n_full else 0.0
    answer_only_acc = correct_answer / n_answer if n_answer else 0.0

    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Data: {data_path.name}  n={len(pairs)}  t_corrupt={t_corrupt}")
    print(f"Full token accuracy (non-pad): {full_acc:.4f}")
    print(f"Answer-only token accuracy:   {answer_only_acc:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
