
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def build_vocab(texts: list[str], min_freq: int = 1) -> tuple[dict[str, int], dict[int, str]]:
    from collections import Counter
    word2id = {"<pad>": 0, "<unk>": 1}
    cnt: Counter = Counter()
    for s in texts:
        for w in re.findall(r"[a-z0-9]+", s.lower()):
            cnt[w] += 1
    for w, c in cnt.most_common():
        if c >= min_freq and w not in word2id:
            word2id[w] = len(word2id)
    id2word = {i: w for w, i in word2id.items()}
    return word2id, id2word


def tokenize(text: str, word2id: dict[str, int], max_len: int) -> list[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    toks = [word2id.get(w, 1) for w in words[:max_len]]
    return toks + [0] * (max_len - len(toks))


def tokenize_qa(
    question: str,
    answer: str,
    word2id: dict[str, int],
    max_q_len: int,
    max_a_len: int,
) -> tuple[list[int], list[float]]:
    q_toks = re.findall(r"[a-z0-9]+", question.lower())[:max_q_len]
    a_toks = re.findall(r"[a-z0-9]+", answer.lower())[:max_a_len]
    q_ids = [word2id.get(w, 1) for w in q_toks] + [0] * (max_q_len - len(q_toks))
    a_ids = [word2id.get(w, 1) for w in a_toks] + [0] * (max_a_len - len(a_toks))
    ids = q_ids + a_ids
    mask = [0.0] * max_q_len + [1.0 if a_ids[i] != 0 else 0.0 for i in range(max_a_len)]
    return ids, mask


def load_csv_texts(path: Path, text_col: str = "text", max_rows: int = 0) -> list[str]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        lower_to_original = {name.lower(): name for name in fieldnames}
        selected_col = lower_to_original.get(text_col.lower(), text_col)
        if selected_col not in fieldnames and fieldnames:
            selected_col = fieldnames[0]
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            t = (row.get(selected_col, "") or "").strip()
            if t:
                rows.append(t)
    return rows


def load_qa_pairs(
    path: Path,
    question_col: str = "question",
    answer_col: str = "answer",
    text_col: str = "text",
    max_rows: int = 0,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            q = (row.get(question_col) or "").strip()
            a = (row.get(answer_col) or "").strip()
            if not q or not a:
                t = (row.get(text_col) or "").strip()
                if t and "?" in t:
                    idx = t.index("?") + 1
                    q, a = t[:idx].strip(), t[idx:].strip()
                elif t:
                    parts = t.rsplit(".", 1)
                    q = (parts[0] + ".").strip() if len(parts) > 1 else ""
                    a = parts[-1].strip() if parts else t
            if q and a:
                pairs.append((q, a))
    return pairs


def batch_iter(
    token_ids: list[list[int]],
    batch_size: int,
    shuffle: bool = True,
) -> list[torch.Tensor]:
    if shuffle:
        import random
        random.shuffle(token_ids)
    batches = []
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i : i + batch_size]
        if len(batch) < batch_size:
            batch = batch + [batch[0]] * (batch_size - len(batch))
        batches.append(torch.tensor(batch, dtype=torch.long))
    return batches


def batch_iter_qa(
    pairs: list[tuple[list[int], list[float]]],
    batch_size: int,
    shuffle: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if shuffle:
        import random
        random.shuffle(pairs)
    batches = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + [chunk[0]] * (batch_size - len(chunk))
        x_0 = torch.tensor([p[0] for p in chunk], dtype=torch.long)
        mask = torch.tensor([p[1] for p in chunk], dtype=torch.float32)
        batches.append((x_0, mask))
    return batches


def main() -> None:
    p = argparse.ArgumentParser(description="Train LODD-SV on CSV data")
    p.add_argument("--data", type=str, default=str(DATA_DIR / "truthful_qa_generation.csv"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--local", action="store_true", default=True, help="Use local predictive coding")
    p.add_argument("--no-local", action="store_false", dest="local")
    p.add_argument("--steps", type=int, default=100, help="Max steps per epoch (0 = full)")
    p.add_argument("--out", type=str, default="", help="Checkpoint path (default: data/lodd_sv_checkpoint.pt)")
    p.add_argument("--d-model", type=int, default=128, help="Model hidden size")
    p.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    p.add_argument("--num-heads", type=int, default=2, help="Number of attention heads")
    p.add_argument("--qa", action="store_true", help="QA mode: loss only on answer tokens (requires question/answer columns)")
    p.add_argument("--max-q-len", type=int, default=0, help="QA: max question length (default: seq_len//2)")
    p.add_argument("--max-a-len", type=int, default=0, help="QA: max answer length (default: seq_len//2)")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print("Run first: python scripts/download_data.py")
        raise SystemExit(1)

    if args.qa:
        pairs = load_qa_pairs(data_path)
        print(f"Loaded {len(pairs)} QA pairs from {data_path}")
        if not pairs:
            raise SystemExit(2)
        texts = [f"{q} {a}" for q, a in pairs]
        word2id, id2word = build_vocab(texts)
        vocab_size = len(word2id)
        print(f"Vocab size: {vocab_size}")
        max_q_len = args.max_q_len or (args.seq_len // 2)
        max_a_len = args.max_a_len or (args.seq_len - max_q_len)
        assert max_q_len + max_a_len <= args.seq_len, "max_q_len + max_a_len must be <= seq_len"
        seq_len = max_q_len + max_a_len
        qa_pairs_tok = [tokenize_qa(q, a, word2id, max_q_len, max_a_len) for q, a in pairs]
        batches = batch_iter_qa(qa_pairs_tok, args.batch)
        if args.steps and args.steps > 0:
            batches = batches[: args.steps]
        print(f"QA: max_q_len={max_q_len} max_a_len={max_a_len}  batches per epoch: {len(batches)}")
    else:
        texts = load_csv_texts(data_path)
        print(f"Loaded {len(texts)} texts from {data_path}")
        if not texts:
            raise SystemExit(2)
        word2id, id2word = build_vocab(texts)
        vocab_size = len(word2id)
        print(f"Vocab size: {vocab_size}")
        seq_len = args.seq_len
        token_ids = [tokenize(t, word2id, seq_len) for t in texts]
        batches = batch_iter(token_ids, args.batch)
        if args.steps and args.steps > 0:
            batches = batches[: args.steps]
        print(f"Batches per epoch: {len(batches)}")

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
    from lodd_sv.local_coding.async_optimizers import LayerWiseOptimizers
    from lodd_sv.training.train import train_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)

    if args.qa and args.local:
        print("QA mode uses answer-only loss; switching to global optimizer (--no-local).")
        args.local = False
    if args.local:
        layer_optimizers = LayerWiseOptimizers(model, lr=1e-4)
        global_optimizer = None
    else:
        layer_optimizers = None
        global_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        def _iter():
            for b in batches:
                if isinstance(b, tuple):
                    yield (b[0].to(device), b[1].to(device))
                else:
                    yield b.to(device)

        metrics = train_epoch(
            model,
            forward_process,
            _iter(),
            use_local_coding=args.local,
            global_optimizer=global_optimizer,
            layer_optimizers=layer_optimizers,
            device=device,
            max_steps=args.steps or None,
        )
        print(f"Epoch {epoch + 1}/{args.epochs}  {metrics}")

    out_ckpt = Path(args.out) if args.out else PROJECT_ROOT / "data" / "lodd_sv_checkpoint.pt"
    if not out_ckpt.is_absolute():
        out_ckpt = PROJECT_ROOT / out_ckpt
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dict = {"model": model.state_dict(), "word2id": word2id, "id2word": id2word}
    ckpt_dict["model_config"] = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_steps": num_steps,
        "seq_len": seq_len,
    }
    if args.qa:
        ckpt_dict["training_mode"] = "qa"
        ckpt_dict["max_q_len"] = max_q_len
        ckpt_dict["max_a_len"] = max_a_len
    else:
        ckpt_dict["training_mode"] = "full"
    torch.save(ckpt_dict, out_ckpt)
    print(f"Checkpoint saved -> {out_ckpt}")


if __name__ == "__main__":
    main()
