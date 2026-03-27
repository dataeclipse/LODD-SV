
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT = DATA_DIR / "lodd_sv_checkpoint.pt"
CSV_FOR_KB = DATA_DIR / "simple_qa_verified.csv"


def _infer_checkpoint_config(ckpt: dict) -> tuple[int, int, int, int]:
    cfg = ckpt.get("model_config", {}) or {}
    state = ckpt["model"]
    d_model = int(cfg.get("d_model") or state["denoising_stack.token_embed.weight"].shape[1])
    num_layers = int(
        cfg.get("num_layers")
        or len(
            {
                k.split(".")[2]
                for k in state.keys()
                if k.startswith("denoising_stack.layers.") and ".self_attn.in_proj_weight" in k
            }
        )
    )
    num_heads = int(cfg.get("num_heads") or 2)
    num_steps = int(cfg.get("num_steps") or state["denoising_stack.time_embed.pe"].shape[0])
    return d_model, num_layers, num_heads, num_steps


def _safe_console_text(text: str) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=str(CKPT), help="Path to .pt checkpoint")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = argmax; >0 = sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=0, help="Override checkpoint hidden size")
    ap.add_argument("--num-layers", type=int, default=0, help="Override checkpoint layers")
    ap.add_argument("--num-heads", type=int, default=0, help="Override checkpoint heads")
    ap.add_argument("--steps", type=int, default=0, help="Denoising steps (0 = checkpoint num_steps)")
    ap.add_argument("--threshold", type=float, default=2.0, help="Router uncertainty threshold (default: 2.0)")
    ap.add_argument("--boost", type=float, default=5.0, help="Logit boost for fact tokens (default: 5.0)")
    ap.add_argument("--debug-router", action="store_true", help="Print when router triggers and which fact was used")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print("No checkpoint. Run: python scripts/train_on_data.py")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    word2id = ckpt["word2id"]
    id2word = ckpt["id2word"]
    vocab_size = len(word2id)
    d_model_ckpt, num_layers_ckpt, num_heads_ckpt, num_steps_ckpt = _infer_checkpoint_config(ckpt)
    d_model = args.d_model or d_model_ckpt
    num_layers = args.num_layers or num_layers_ckpt
    num_heads = args.num_heads or num_heads_ckpt
    model_num_steps = num_steps_ckpt
    sampling_steps = args.steps or model_num_steps
    if sampling_steps > model_num_steps:
        print(
            f"Requested --steps={sampling_steps} exceeds checkpoint num_steps={model_num_steps}; using {model_num_steps}."
        )
        sampling_steps = model_num_steps


    from lodd_sv.verification import InMemoryKnowledgeBase, StatisticalRouter
    kb = InMemoryKnowledgeBase()
    if CSV_FOR_KB.exists():
        with open(CSV_FOR_KB, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = (row.get("text", "") or f"{row.get('question','')} {row.get('answer','')}").strip()
                if text:
                    kb.add(f"f{i}", text)
        print(f"Loaded {len(kb._store)} facts into KB from {CSV_FOR_KB.name}")
    router = StatisticalRouter(kb, threshold=args.threshold)

    from lodd_sv.math.state_space import VocabularyStateSpace
    from lodd_sv.engine.forward_process import ForwardProcess
    from lodd_sv.engine.reverse_network import ReverseDenoisingNetwork

    state_space = VocabularyStateSpace(vocab_size=vocab_size, token_to_id=word2id, id_to_token=id2word)
    forward_process = ForwardProcess(state_space, num_steps=model_num_steps, device=device)
    model = ReverseDenoisingNetwork(
        state_space,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_steps=model_num_steps,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    num_samples = 2
    seq_len = 32
    x_t = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)

    def tokenize_fact(fact: str):
        out = []
        for w in re.findall(r"[a-z0-9]+", fact.lower()):
            tid = word2id.get(w, 1)
            if tid != 0:
                out.append(tid)
        return out[:seq_len]

    with torch.no_grad():
        gate_trigger_count = 0
        retrieved_count = 0
        injected_count = 0

        u_vals = []
        for step in range(sampling_steps, 0, -1):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            logits, _ = model(x_t, t)
            probs = torch.softmax(logits, dim=-1)
            for b in range(num_samples):
                r = router(probs[b : b + 1], x_t[b : b + 1], id_to_token=id2word.get)
                u_vals.append((step, b, r.uncertainty_value))
                if r.triggered:
                    gate_trigger_count += 1
                    if args.debug_router:
                        q_snip = (r.query_used or "").replace("\n", " ")[:80]
                        print(
                            _safe_console_text(
                                f"[router] step={step} sample={b} triggered=True u={r.uncertainty_value:.4f} q='{q_snip}'"
                            )
                        )
                if r.triggered and r.retrieved_fact:
                    retrieved_count += 1
                    fact_tok = tokenize_fact(r.retrieved_fact)
                    if fact_tok:
                        injected_count += 1
                        if args.debug_router:
                            snippet = " ".join(r.retrieved_fact.split()[:18])
                            print(
                                _safe_console_text(
                                    f"[router] step={step} sample={b} triggered=True boost={args.boost} "
                                    f"fact_tokens={len(fact_tok)} fact='{snippet}{'...' if len(snippet) < len(r.retrieved_fact) else ''}'"
                                )
                            )
                        logits_b = router.inject_fact_into_logits(
                            logits[b : b + 1], fact_tok, 0, boost=float(args.boost)
                        )
                        logits = logits.clone()
                        logits[b : b + 1] = logits_b
            if args.temperature <= 0:
                x_t = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                x_t = torch.multinomial(probs.view(-1, vocab_size), 1).view(num_samples, seq_len)
        if args.debug_router:
            print(f"[router] gate_triggers={gate_trigger_count} retrieved={retrieved_count} injections={injected_count}")

            try:
                vals_only = [v for (_, _, v) in u_vals]
                finite = [v for v in vals_only if float("-inf") < v < float("inf")]
                if finite:
                    print(f"[router] uncertainty finite: min={min(finite):.4f} max={max(finite):.4f} threshold={args.threshold}")
                else:
                    print(f"[router] uncertainty had no finite values; threshold={args.threshold}")
            except Exception:
                pass

    for i in range(num_samples):
        tokens = x_t[i].cpu().tolist()
        words = [id2word.get(j, "<unk>") for j in tokens if j != 0]
        text = " ".join(w for w in words if w != "<pad>").strip()
        print(f"[{i+1}] {text[:250]}{'...' if len(text) > 250 else ''}")
    print("Done.")


if __name__ == "__main__":
    main()
