
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def save_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def download_truthful_qa() -> None:
    from datasets import load_dataset

    ds_gen = load_dataset("truthfulqa/truthful_qa", "generation")
    rows = []
    for ex in ds_gen["validation"]:
        q = ex.get("question", "")
        best = ex.get("best_answer") or ex.get("correct_answers", [""])
        ans = best[0] if isinstance(best, list) else best
        rows.append({"question": q, "answer": ans, "text": f"{q} {ans}"})
    out = DATA_DIR / "truthful_qa_generation.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved TruthfulQA generation {len(rows)} rows -> {out}")


    ds_mc = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    rows_mc = []
    for ex in ds_mc["validation"]:
        q = ex.get("question", "")
        mc1 = ex.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        idx = next((i for i, L in enumerate(labels) if L == 1), 0)
        ans = choices[idx] if idx < len(choices) else ""
        rows_mc.append({"question": q, "answer": ans, "text": f"{q} {ans}"})
    out_mc = DATA_DIR / "truthful_qa_multiple_choice.csv"
    save_csv(out_mc, rows_mc, ["question", "answer", "text"])
    print(f"Saved TruthfulQA multiple_choice {len(rows_mc)} rows -> {out_mc}")


def download_simple_qa() -> None:
    from datasets import load_dataset
    ds = load_dataset("OpenEvals/SimpleQA")
    rows = []
    for split in ds:
        for ex in ds[split]:
            q = ex.get("question", ex.get("Question", ""))
            a = ex.get("answer", ex.get("Answer", ""))
            if isinstance(a, list):
                a = a[0] if a else ""
            rows.append({"question": q, "answer": a, "text": f"{q} {a}"})
    out = DATA_DIR / "simple_qa_train.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved OpenEvals/SimpleQA {len(rows)} rows -> {out}")


def download_trivia_tiny() -> None:
    from datasets import load_dataset
    ds = load_dataset("cedricmkl/triviaqa-tiny-5k")
    rows = []
    for split in ds:
        for ex in ds[split]:
            q = ex.get("question", ex.get("Question", ""))
            a = ex.get("answer", ex.get("Answer", ""))
            if isinstance(a, dict):
                a = a.get("value", a.get("aliases", [""])[0] if a.get("aliases") else "")
            elif isinstance(a, list):
                a = a[0] if a else ""
            rows.append({"question": q, "answer": a, "text": f"{q} {a}"})
    out = DATA_DIR / "triviaqa_tiny_train.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved cedricmkl/triviaqa-tiny-5k {len(rows)} rows -> {out}")


def download_simple_qa_verified() -> None:
    from datasets import load_dataset
    ds = load_dataset("codelion/SimpleQA-Verified")
    rows = []
    for split in ds:
        for ex in ds[split]:
            q = ex.get("question", ex.get("Question", ""))
            a = ex.get("answer", ex.get("Answer", ""))
            if isinstance(a, list):
                a = a[0] if a else ""
            rows.append({"question": q, "answer": a, "text": f"{q} {a}"})
    out = DATA_DIR / "simple_qa_verified.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved codelion/SimpleQA-Verified {len(rows)} rows -> {out}")

def download_sentence_transformers_nq(*, max_rows: int = 0) -> None:
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/natural-questions")
    rows = []
    for i, ex in enumerate(ds["train"]):
        if max_rows and i >= max_rows:
            break
        q = (ex.get("query") or "").strip()
        a = (ex.get("answer") or "").strip()
        if q and a:
            rows.append({"question": q, "answer": a, "text": f"{q} {a}"})
    out = DATA_DIR / "natural_questions_sentence_transformers.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved sentence-transformers/natural-questions {len(rows)} rows -> {out}")


def download_google_natural_questions(*, config: str, split: str = "train", max_rows: int = 0) -> None:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/natural_questions", config)
    if split not in ds:

        split = next(iter(ds.keys()))

    def _first_string(x) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, list) and x:
            for it in x:
                s = _first_string(it)
                if s:
                    return s
        if isinstance(x, dict):

            for k in ("text", "value", "answer", "short_answer", "long_answer"):
                s = _first_string(x.get(k))
                if s:
                    return s
        return ""

    rows = []
    for i, ex in enumerate(ds[split]):
        if max_rows and i >= max_rows:
            break
        q = (ex.get("question") or ex.get("query") or "").strip()

        a = ""
        if "answer" in ex:
            a = _first_string(ex.get("answer"))
        if not a and "answers" in ex:
            a = _first_string(ex.get("answers"))
        if not a and "annotations" in ex:
            a = _first_string(ex.get("annotations"))
        if not a and "short_answers" in ex:
            a = _first_string(ex.get("short_answers"))
        if not a and "long_answer" in ex:
            a = _first_string(ex.get("long_answer"))

        if q and a:
            rows.append({"question": q, "answer": a, "text": f"{q} {a}"})

    safe_cfg = re.sub(r"[^a-z0-9_]+", "_", config.lower())
    safe_split = re.sub(r"[^a-z0-9_]+", "_", split.lower())
    out = DATA_DIR / f"natural_questions_google_{safe_cfg}_{safe_split}.csv"
    save_csv(out, rows, ["question", "answer", "text"])
    print(f"Saved google NQ ({config}/{split}) {len(rows)} rows -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--include-sentence-transformers-nq", action="store_true", help="Download sentence-transformers/natural-questions")
    p.add_argument("--include-google-nq", action="store_true", help="Download google-research-datasets/natural_questions (LARGE)")
    p.add_argument("--google-nq-configs", type=str, default="default,dev", help="Comma-separated configs for google NQ")
    p.add_argument("--google-nq-split", type=str, default="train", help="Split to export (train/validation/test if available)")
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows per dataset export (0 = no limit)")
    args = p.parse_args()

    print("Downloading datasets into", DATA_DIR)
    download_truthful_qa()
    try:
        download_simple_qa()
    except Exception as e:
        print("OpenEvals/SimpleQA skip:", e)
    try:
        download_trivia_tiny()
    except Exception as e:
        print("TriviaQA tiny skip:", e)
    try:
        download_simple_qa_verified()
    except Exception as e:
        print("SimpleQA-Verified skip:", e)

    if args.include_sentence_transformers_nq:
        try:
            download_sentence_transformers_nq(max_rows=args.max_rows)
        except Exception as e:
            print("sentence-transformers/natural-questions skip:", e)

    if args.include_google_nq:
        import re
        cfgs = [c.strip() for c in (args.google_nq_configs or "").split(",") if c.strip()]
        if not cfgs:
            cfgs = ["default"]
        for cfg in cfgs:
            try:
                download_google_natural_questions(config=cfg, split=args.google_nq_split, max_rows=args.max_rows)
            except Exception as e:
                print(f"google NQ ({cfg}) skip:", e)

    print("Done. Use data/*.csv for training and KB.")


if __name__ == "__main__":
    main()
