
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_qa_pairs(path: Path, max_rows: int = 0):
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            t = (row.get("text") or "").strip()
            if not q or not a:
                if t and "?" in t:
                    idx = t.index("?") + 1
                    q, a = t[:idx].strip(), t[idx:].strip()
                elif t and not a:
                    a = t
                elif t and not q:
                    q = ""
            if a:
                pairs.append((q or "", a))
    return pairs


def tokenize_words(text: str, word2id: dict, max_len: int) -> list[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_len]
    return [word2id.get(w, 1) for w in words] + [0] * (max_len - len(words))


def sample_x_prev_from_posterior(forward, x_t_cur: torch.Tensor, x0_hat: torch.Tensor, t_int: int, vocab_size: int, device) -> torch.Tensor:

    return x0_hat


def run_reverse(
    model,
    forward,
    x_t: torch.Tensor,
    t_start: int,
    steps: int,
    temperature: float,
    vocab_size: int,
    device,
    router=None,
    word2id=None,
    id2word=None,
    boost: float = 20.0,
) -> torch.Tensor:
    x_cur = x_t.clone()
    for step in range(t_start, max(1, t_start - steps), -1):
        t = torch.full((x_cur.size(0),), step, device=device, dtype=torch.long)
        logits, _ = model(x_cur, t)
        if router is not None and word2id is not None and id2word is not None:
            probs = torch.softmax(logits, dim=-1)
            id_to_token = lambda i: id2word.get(i, "<unk>")
            for b in range(x_cur.size(0)):
                r = router(probs[b : b + 1], x_cur[b : b + 1], id_to_token=id_to_token)
                if r.triggered and r.retrieved_fact:
                    fact_tok = [word2id.get(w, 1) for w in re.findall(r"[a-z0-9]+", r.retrieved_fact.lower()) if word2id.get(w, 1) != 0][:32]
                    if fact_tok:
                        logits = logits.clone()
                        logits_b = router.inject_fact_into_logits(logits[b : b + 1], fact_tok, 0, boost=boost)
                        logits[b : b + 1] = logits_b
        if temperature <= 0:
            x0_hat = logits.argmax(dim=-1)
        else:
            probs0 = torch.softmax(logits / temperature, dim=-1)
            x0_hat = torch.multinomial(probs0.view(-1, vocab_size), 1).view(x_cur.size(0), x_cur.size(1))
        x_cur = sample_x_prev_from_posterior(forward, x_cur, x0_hat, step, vocab_size, device)
    return x_cur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DATA_DIR / "ckpt_global_large_combined_tok_25ep.pt"))
    p.add_argument("--data", type=str, default=str(DATA_DIR / "simple_qa_verified.csv"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--t-corrupt", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--boost", type=float, default=20.0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=32)
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

    pairs = load_qa_pairs(data_path, max_rows=args.n)
    if not pairs:
        print("No QA pairs.")
        raise SystemExit(2)

    max_q_len = args.seq_len // 2
    max_a_len = args.seq_len - max_q_len
    refs = []
    answer_masks = []
    for q, a in pairs:
        q_toks = tokenize_words(q, word2id, max_q_len)
        a_toks = tokenize_words(a, word2id, max_a_len)
        seq = q_toks + a_toks
        refs.append(seq)
        mask = [0.0] * max_q_len + [1.0 if a_toks[i] != 0 else 0.0 for i in range(max_a_len)]
        answer_masks.append(mask)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork
    from lodd_sv.verification import InMemoryKnowledgeBase, StatisticalRouter

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

    kb = InMemoryKnowledgeBase()
    with open(data_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            text = (row.get("text", "") or f"{row.get('question','')} {row.get('answer','')}").strip()
            if text:
                kb.add(f"f{i}", text)
    router = StatisticalRouter(kb, threshold=args.threshold)

    ref = torch.tensor(refs, dtype=torch.long, device=device)
    answer_mask_t = torch.tensor(answer_masks, device=device, dtype=torch.float)
    t_corrupt = max(1, min(args.t_corrupt, num_steps))
    t_val = torch.full((ref.size(0),), t_corrupt, device=device, dtype=torch.long)
    x_t, _ = forward.corrupt(ref, t_val)

    with torch.no_grad():
        pred_wo = run_reverse(
            model, forward, x_t, t_corrupt, args.steps, args.temperature,
            vocab_size, device, router=None,
        )
        pred_w = run_reverse(
            model, forward, x_t, t_corrupt, args.steps, args.temperature,
            vocab_size, device, router=router, word2id=word2id, id2word=id2word, boost=args.boost,
        )

    def answer_acc(pred: torch.Tensor) -> float:
        match = (pred == ref).float()
        n = answer_mask_t.sum().item()
        if n == 0:
            return 0.0
        return (match * answer_mask_t).sum().item() / n

    acc_wo = answer_acc(pred_wo)
    acc_w = answer_acc(pred_w)
    delta = acc_w - acc_wo

    print("Router impact (multi-step reverse, answer-only token accuracy)")
    print("Checkpoint:", ckpt_path.name)
    print("Data:", data_path.name, " n=", len(pairs), " t_corrupt=", t_corrupt, " steps=", args.steps)
    print("Without router (answer-only acc):", f"{acc_wo:.4f}")
    print("With router (answer-only acc):   ", f"{acc_w:.4f}")
    print("Delta (with - without):          ", f"{delta:+.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
