# LODD-SV

**by [@dataeclipse](https://github.com/dataeclipse)** — Locally-Optimized Discrete diffusion with Symbolic verification.

This repository implements:
- discrete diffusion for text (`x0 -> xt -> x0`),
- local predictive coding training (layer-wise),
- global cross-entropy training,
- symbolic routing with a fact knowledge base at inference.

## Features

- Word-level tokenization and vocabulary build from CSV data.
- Two training modes:
  - local (`--local`, default),
  - global (`--no-local`).
- QA mode (`--qa`) with answer-only loss mask.
- Checkpoint-based generation and reconstruction evaluation.
- Router-based generation with uncertainty-triggered fact injection.
- Reproducible scripts for ablations and evaluation harnesses.

## Project Layout

```text
data-science/
  lodd_sv/
    math/
    engine/
    local_coding/
    verification/
    training/
  scripts/
    download_data.py
    build_combined_csv.py
    train_on_data.py
    eval_checkpoint.py
    eval_qa.py
    generate_from_checkpoint.py
    generate_conditioned.py
    generate_with_router.py
    eval_router_impact.py
    run_eval_harness.py
    run_ablations.py
    run_all.py
  data/
  tests/
  run_demo.py
```

## Requirements

- Python 3.9+
- PyTorch 2.0+

Install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run demo:

```bash
python run_demo.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Data

Download datasets into `data/`:

```bash
python scripts/download_data.py
```

Build combined CSV:

```bash
python scripts/build_combined_csv.py
```

## Training

Global training (recommended for non-zero reconstruction accuracy):

```bash
python scripts/train_on_data.py --data data/simple_qa_verified.csv --epochs 5 --no-local --steps 0 --out data/ckpt_global.pt
```

Local training:

```bash
python scripts/train_on_data.py --data data/simple_qa_verified.csv --epochs 5 --local --steps 0 --out data/ckpt_local.pt
```

QA training (answer-only loss):

```bash
python scripts/train_on_data.py --data data/truthful_qa_generation.csv --qa --no-local --epochs 5 --seq_len 32 --out data/ckpt_qa.pt
```

## Evaluation

Reconstruction evaluation:

```bash
python scripts/eval_checkpoint.py --checkpoint data/ckpt_global.pt --data data/simple_qa_verified.csv --mode reconstruct --t-corrupt 5 --n 100
```

QA evaluation:

```bash
python scripts/eval_qa.py --checkpoint data/ckpt_qa.pt --data data/truthful_qa_generation.csv --t-corrupt 5 --n 50
```

Evaluation harness:

```bash
python scripts/run_eval_harness.py --data data/simple_qa_verified.csv --n 50
```

## Generation

From checkpoint:

```bash
python scripts/generate_from_checkpoint.py --checkpoint data/ckpt_global.pt --steps 50 --temperature 0.8
```

Conditioned generation:

```bash
python scripts/generate_conditioned.py --checkpoint data/ckpt_global.pt --prompt "What is the capital of Ghana?" --steps 20 --temperature 0.8
```

With symbolic router:

```bash
python scripts/generate_with_router.py --checkpoint data/ckpt_global.pt --steps 20 --temperature 0.8 --threshold 2.0 --boost 5.0
```

## Ablations and Full Pipeline

Ablation run:

```bash
python scripts/run_ablations.py --data data/simple_qa_verified.csv --epochs 3 --n 50
```

End-to-end run:

```bash
python scripts/run_all.py
```

## Checkpoints

`.pt` checkpoint files are not committed (large binaries; one file exceeded GitHub's 100 MB limit). Train with the commands above or use your own paths.

Checkpoint files store:
- model weights,
- `word2id` and `id2word`,
- `training_mode`,
- `model_config` (`d_model`, `num_layers`, `num_heads`, `num_steps`, `seq_len`).

## Notes

- For reconstruction quality, use global mode (`--no-local`).
- Local-only mode is useful for representation experiments but often low on reconstruction metrics.
- For larger checkpoints, pass matching model args when a script requires them.

## License

MIT
