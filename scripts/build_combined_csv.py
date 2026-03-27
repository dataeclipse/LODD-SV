
from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def row_to_text(row: dict) -> str:
    return (row.get("text") or f"{row.get('question','')} {row.get('answer','')}").strip()


def main() -> None:
    candidates = [
        DATA_DIR / "simple_qa_verified.csv",
        DATA_DIR / "simple_qa_train.csv",
        DATA_DIR / "truthfulqa_generation.csv",
        DATA_DIR / "truthfulqa_multiple_choice.csv",
        DATA_DIR / "triviaqa_tiny.csv",
    ]

    texts: list[str] = []
    used = []
    for p in candidates:
        if not p.exists():
            continue
        used.append(p.name)
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row_to_text(row)
                if t:
                    texts.append(t)

    out = DATA_DIR / "combined_train.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text"])
        w.writeheader()
        for t in texts:
            w.writerow({"text": t})

    print("Inputs:", ", ".join(used) if used else "(none found)")
    print("Wrote:", out)
    print("Rows:", len(texts))


if __name__ == "__main__":
    main()

