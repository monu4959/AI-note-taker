"""Step 1.3: build the hand-label sheet that validates the verifier and the judge.

Part A (25 rows): our items with cited lines -> you label correct / partly / wrong.
Part B (15 rows): judge matches (AMI sentence + our items) -> you label covered / partial / missed.
Stratified, fixed seed, test meetings only. Writes evaluation/labels/step1_labels.csv; your
answers go in the `your_label` column. The model's labels are in a separate file so they
don't anchor your judgement.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ami_summaries import human_summary
from pipeline import ami
from pipeline.insights.verify import render

EVAL = ami.ROOT / "artifacts" / "eval"
OUT = ami.ROOT / "evaluation" / "labels"
SEED = 17


def main() -> None:
    run = json.loads((EVAL / "runs" / "test-p2.json").read_text())
    rng = random.Random(SEED)
    pool_a, pool_b = [], []
    for m in run["meetings"]:
        ins_path = sorted((EVAL / "asr" / m).glob("insights.*.json"),
                          key=lambda p: p.stat().st_mtime)[-1]
        insights = json.loads(ins_path.read_text())
        fp = ins_path.name.split(".")[1]
        utts = {u["id"]: u for u in json.loads(
            (ami.ROOT / "artifacts" / "meetings" / f"ami-{m}" / "assign" / "utterances.json").read_text())}
        items = {i["id"]: i for i in insights["items"]}
        for it in insights["items"]:
            if it["type"] in ("decision", "action", "metric") and it["verification"]["status"] != "rejected":
                cited = " || ".join(f"[{u}] {utts[u]['speaker']}: {utts[u]['text']}"
                                    for u in it["evidence"]["utterance_ids"] if u in utts)
                pool_a.append({"meeting": m, "item_id": it["id"], "type": it["type"],
                               "status": it["verification"]["status"], "item": render(it),
                               "cited_lines": cited})
        human = {h["id"]: h for h in human_summary(m)}
        judge = json.loads(next((EVAL / "asr" / m).glob(f"judge.*.{fp}.json")).read_text())
        for j in judge:
            if j["human_id"] in human:
                ours = " || ".join(f"{i}: {render(items[i])}" for i in j["item_ids"] if i in items)
                pool_b.append({"meeting": m, "human_id": j["human_id"],
                               "section": human[j["human_id"]]["section"],
                               "human_sentence": human[j["human_id"]]["text"],
                               "our_items": ours or "(none)", "judge": j["status"]})

    # Part A: 10 decisions, 8 actions, 7 metrics; ~1/3 needs_review within each type
    sample_a = []
    for kind, k in (("decision", 10), ("action", 8), ("metric", 7)):
        rows = [r for r in pool_a if r["type"] == kind]
        nr = [r for r in rows if r["status"] == "needs_review"]
        ok = [r for r in rows if r["status"] == "verified"]
        n_nr = min(len(nr), k // 3)
        sample_a += rng.sample(nr, n_nr) + rng.sample(ok, k - n_nr)
    # Part B: 15 judge matches, spread over statuses (covered 9, partial 4, missed 2 where possible)
    sample_b = []
    for status, k in (("covered", 9), ("partial", 4), ("missed", 2)):
        rows = [r for r in pool_b if r["judge"] == status]
        sample_b += rng.sample(rows, min(k, len(rows)))
    rng.shuffle(sample_a)
    rng.shuffle(sample_b)

    OUT.mkdir(parents=True, exist_ok=True)
    sheet = OUT / "step1_labels.csv"
    key = OUT / "step1_labels_model.json"
    with sheet.open("w", newline="", encoding="utf-8-sig") as fh:  # BOM so Excel reads UTF-8
        w = csv.writer(fh)
        w.writerow(["row", "part", "your_label", "allowed_labels", "what_to_judge",
                    "meeting", "text_1", "text_2"])
        for i, r in enumerate(sample_a, 1):
            w.writerow([f"A{i}", "A", "", "correct / partly / wrong",
                        "Is the item a correct note of what the cited lines say?",
                        r["meeting"], r["item"], r["cited_lines"]])
        for i, r in enumerate(sample_b, 1):
            w.writerow([f"B{i}", "B", "", "covered / partial / missed",
                        "Do our items (text_2) convey the human summary sentence (text_1)?",
                        r["meeting"], f"[{r['section']}] {r['human_sentence']}", r["our_items"]])
    model = {f"A{i}": {"meeting": r["meeting"], "item_id": r["item_id"], "type": r["type"],
                       "verifier_status": r["status"]} for i, r in enumerate(sample_a, 1)}
    model |= {f"B{i}": {"meeting": r["meeting"], "human_id": r["human_id"],
                        "judge": r["judge"]} for i, r in enumerate(sample_b, 1)}
    key.write_text(json.dumps(model, indent=1))
    print(f"{len(sample_a)} item rows + {len(sample_b)} judge rows -> {sheet}")


if __name__ == "__main__":
    main()
