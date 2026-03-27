
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def run_eval_checkpoint(ckpt: str, data: str, t: int, n: int, d_model: int, num_layers: int, num_heads: int) -> str | None:
    cmd = [
        sys.executable,
        "scripts/eval_checkpoint.py",
        "--checkpoint", ckpt,
        "--data", data,
        "--mode", "reconstruct",
        "--t-corrupt", str(t),
        "--n", str(n),
    ]
    if d_model != 128 or num_layers != 2 or num_heads != 2:
        cmd += ["--d-model", str(d_model), "--num-layers", str(num_layers), "--num-heads", str(num_heads)]
    out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if "Token accuracy:" in line:
            return line.split(":", 1)[1].strip()
    return None


def run_eval_qa(ckpt: str, data: str, t: int, n: int, d_model: int, num_layers: int, num_heads: int) -> tuple[str | None, str | None]:
    cmd = [
        sys.executable,
        "scripts/eval_qa.py",
        "--checkpoint", ckpt,
        "--data", data,
        "--t-corrupt", str(t),
        "--n", str(n),
        "--d-model", str(d_model),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
    ]
    out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return None, None
    full_acc = answer_acc = None
    for line in out.stdout.splitlines():
        if "Full token accuracy" in line:
            full_acc = line.split(":", 1)[1].strip()
        if "Answer-only token accuracy" in line:
            answer_acc = line.split(":", 1)[1].strip()
    return full_acc, answer_acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DATA_DIR / "simple_qa_verified.csv"))
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print("Data not found:", data_path)
        raise SystemExit(1)

    checkpoints = [
        ("data/ckpt_global.pt", 128, 2, 2),
        ("data/ckpt_global_20ep.pt", 128, 2, 2),
        ("data/ckpt_global_large.pt", 256, 3, 4),
        ("data/ckpt_global_large_25ep.pt", 256, 3, 4),
        ("data/ckpt_global_large_combined_tok_25ep.pt", 256, 3, 4),
        ("data/ckpt_qa_1ep.pt", 128, 2, 2),
    ]
    data_str = str(data_path)

    print("Eval harness: reconstruct (t=5, t=10) + QA metrics")
    print("Data:", data_path.name, "n=", args.n)
    print("-" * 72)

    rows = []
    for ckpt, dm, nl, nh in checkpoints:
        if not (PROJECT_ROOT / ckpt).exists():
            rows.append((ckpt, "N/A", "N/A", "N/A", "N/A"))
            continue
        r5 = run_eval_checkpoint(ckpt, data_str, 5, args.n, dm, nl, nh)
        r10 = run_eval_checkpoint(ckpt, data_str, 10, args.n, dm, nl, nh)
        full_acc, answer_acc = run_eval_qa(ckpt, data_str, 5, args.n, dm, nl, nh)
        rows.append((ckpt, r5 or "N/A", r10 or "N/A", full_acc or "N/A", answer_acc or "N/A"))

    print(f"{'Checkpoint':<45} {'t=5':<10} {'t=10':<10} {'full_acc':<10} {'ans_only':<10}")
    for ckpt, r5, r10, full, ans in rows:
        print(f"{ckpt:<45} {r5:<10} {r10:<10} {full:<10} {ans:<10}")
    print("Done.")


if __name__ == "__main__":
    main()
