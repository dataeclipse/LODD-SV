
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"Failed: {desc}", file=sys.stderr)
        return False
    return True


def main() -> None:
    py = sys.executable
    data_csv = str(DATA_DIR / "simple_qa_verified.csv")


    if not run(
        [py, "scripts/train_on_data.py", "--data", data_csv, "--epochs", "20", "--no-local",
         "--steps", "0", "--out", "data/ckpt_global_20ep.pt"],
        "Train global 20 epochs (full data)",
    ):
        sys.exit(1)


    if not run(
        [py, "scripts/train_on_data.py", "--data", data_csv, "--epochs", "15", "--no-local",
         "--steps", "0", "--d-model", "256", "--num-layers", "3", "--num-heads", "4",
         "--batch", "4", "--out", "data/ckpt_global_large.pt"],
        "Train global 15 epochs, d_model=256, num_layers=3",
    ):
        sys.exit(1)


    checkpoints = [
        ("data/ckpt_global.pt", None),
        ("data/ckpt_global_20ep.pt", None),
        ("data/ckpt_global_large.pt", {"d_model": 256, "num_layers": 3, "num_heads": 4}),
    ]
    results = []
    for ckpt, cfg in checkpoints:
        if not (PROJECT_ROOT / ckpt).exists():
            results.append((ckpt, None, None))
            continue
        extra = []
        if cfg:
            extra = ["--d-model", str(cfg["d_model"]), "--num-layers", str(cfg["num_layers"]), "--num-heads", str(cfg["num_heads"])]
        for t in (5, 10):
            out = subprocess.run(
                [py, "scripts/eval_checkpoint.py", "--checkpoint", ckpt,
                 "--data", data_csv, "--mode", "reconstruct", "--t-corrupt", str(t), "--n", "100"] + extra,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            acc = None
            if out.returncode == 0 and "Token accuracy:" in out.stdout:
                for line in out.stdout.splitlines():
                    if line.strip().startswith("Token accuracy:"):
                        acc = line.split(":", 1)[1].strip()
                        break
            results.append((ckpt, t, acc))


    print("\n" + "=" * 60)
    print("RECONSTRUCTION TOKEN ACCURACY SUMMARY (n=100)")
    print("=" * 60)
    for ckpt, _ in checkpoints:
        row = [ckpt]
        for t in (5, 10):
            acc = next((r[2] for r in results if r[0] == ckpt and r[1] == t), None)
            row.append(acc or "N/A")
        print(f"  {row[0]}\n    t=5: {row[1]}\n    t=10: {row[2]}")
    print("Done.")


if __name__ == "__main__":
    main()
