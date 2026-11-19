"""Score the filled hand-label sheet: human precision, and agreement of verifier / judge with you.

Usage: .venv/Scripts/python scripts/step1_score_labels.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "evaluation" / "labels"


def main() -> None:
    rows = list(csv.DictReader((LABELS / "step1_labels.csv").open(encoding="utf-8-sig")))
    model = json.loads((LABELS / "step1_labels_model.json").read_text())
    a = [(r["your_label"].strip().lower(), model[r["row"]]) for r in rows if r["part"] == "A"]
    b = [(r["your_label"].strip().lower(), model[r["row"]]) for r in rows if r["part"] == "B"]
    missing = [r["row"] for r in rows if not r["your_label"].strip()]
    if missing:
        sys.exit(f"{len(missing)} rows still unlabelled: {', '.join(missing)}")

    human = Counter(lbl for lbl, _ in a)
    print(f"Part A, {len(a)} notes: correct {human['correct']}, partly {human['partly']}, wrong {human['wrong']}")
    print(f"  human precision strict {human['correct'] / len(a):.1%}, "
          f"lenient {(human['correct'] + human['partly']) / len(a):.1%}")
    for status in ("verified", "needs_review"):
        sub = [lbl for lbl, m in a if m["verifier_status"] == status]
        if sub:
            c = Counter(sub)
            print(f"  verifier '{status}' ({len(sub)}): correct {c['correct']}, partly {c['partly']}, wrong {c['wrong']}")

    agree = sum(lbl == m["judge"] for lbl, m in b)
    print(f"Part B, {len(b)} judge matches: you agree with the judge on {agree} ({agree / len(b):.1%})")
    for lbl, m in b:
        if lbl != m["judge"]:
            print(f"  disagreement: you {lbl!r}, judge {m['judge']!r} ({m['meeting']} {m['human_id']})")


if __name__ == "__main__":
    main()
