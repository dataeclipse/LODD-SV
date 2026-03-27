
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def run_train(
    data: str,
    epochs: int,
    out: str,
    no_local: bool = False,
    steps_per_epoch: int = 0,
    d_model: int = 128,
    num_layers: int = 2,
    num_heads: int = 2,
    batch: int = 8,
) -> bool:
    cmd = [
        sys.executable,
        "scripts/train_on_data.py",
        "--data", data,
        "--epochs", str(epochs),
        "--out", out,
        "--steps", str(steps_per_epoch),
        "--d-model", str(d_model),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--batch", str(batch),
    ]
    if no_local:
        cmd.append("--no-local")
    r = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return r.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation: train local / global / hybrid, then eval harness")
    p.add_argument("--data", type=str, default=str(DATA_DIR / "simple_qa_verified.csv"))
    p.add_argument("--epochs", type=int, default=3, help="Epochs per run")
    p.add_argument("--steps", type=int, default=0, help="Max steps per epoch (0 = full)")
    p.add_argument("--skip-train", action="store_true", help="Skip training; only run eval harness on existing ckpts")
    p.add_argument("--n", type=int, default=50, help="n samples for eval harness")
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        print("Data not found:", data_path)
        raise SystemExit(1)
    data_str = str(data_path)

    out_dir = PROJECT_ROOT / "data"
    ckpt_local = out_dir / "ckpt_ablation_local.pt"
    ckpt_global = out_dir / "ckpt_ablation_global.pt"
    ckpt_hybrid = out_dir / "ckpt_ablation_hybrid.pt"

    if not args.skip_train:
        print("Ablation 1/3: local-only")
        if not run_train(data_str, args.epochs, str(ckpt_local), no_local=False, steps_per_epoch=args.steps):
            print("Local training failed.")
            raise SystemExit(2)
        print("Ablation 2/3: global-only")
        if not run_train(data_str, args.epochs, str(ckpt_global), no_local=True, steps_per_epoch=args.steps):
            print("Global training failed.")
            raise SystemExit(3)
        print("Ablation 3/3: hybrid (global 2 ep; true hybrid would need --resume in train_on_data)")
        if not run_train(data_str, 2, str(ckpt_hybrid), no_local=True, steps_per_epoch=args.steps):
            print("Hybrid training failed.")
            raise SystemExit(4)
        print("Ablation training done.")
    else:
        print("Skipping training (--skip-train). Using existing ablation checkpoints if present.")


    print("\nEval harness (ablation checkpoints):")
    print("-" * 72)
    rows = []
    for label, ckpt in [
        ("local", ckpt_local),
        ("global", ckpt_global),
        ("hybrid", ckpt_hybrid),
    ]:
        if not ckpt.exists():
            rows.append((label, "N/A", "N/A", "N/A", "N/A"))
            continue
        r5 = subprocess.run(
            [sys.executable, "scripts/eval_checkpoint.py", "--checkpoint", str(ckpt),
             "--data", data_str, "--mode", "reconstruct", "--t-corrupt", "5", "--n", str(args.n)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        r10 = subprocess.run(
            [sys.executable, "scripts/eval_checkpoint.py", "--checkpoint", str(ckpt),
             "--data", data_str, "--mode", "reconstruct", "--t-corrupt", "10", "--n", str(args.n)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        qa = subprocess.run(
            [sys.executable, "scripts/eval_qa.py", "--checkpoint", str(ckpt),
             "--data", data_str, "--t-corrupt", "5", "--n", str(args.n)],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        def acc(out, key):
            if out.returncode != 0:
                return "N/A"
            for line in out.stdout.splitlines():
                if key in line and ":" in line:
                    return line.split(":", 1)[1].strip()
            return "N/A"
        full_acc = answer_acc = "N/A"
        for line in (qa.stdout or "").splitlines():
            if "Full token accuracy" in line:
                full_acc = line.split(":", 1)[1].strip()
            if "Answer-only token accuracy" in line:
                answer_acc = line.split(":", 1)[1].strip()
        rows.append((
            label,
            acc(r5, "Token accuracy"),
            acc(r10, "Token accuracy"),
            full_acc,
            answer_acc,
        ))
    print(f"{'Mode':<10} {'t=5':<10} {'t=10':<10} {'full_acc':<12} {'ans_only':<10}")
    for label, r5, r10, full, ans in rows:
        print(f"{label:<10} {r5:<10} {r10:<10} {full:<12} {ans:<10}")
    print("Done.")


if __name__ == "__main__":
    main()
