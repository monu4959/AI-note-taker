"""Step 4: can the free local judge stand in for the Sonnet judge in the notes evaluation?

Re-judges the cached notes that the Sonnet judge already scored (artifacts/eval/{asr,ref}/
<meeting>/insights.*.json with judge.j1.*.json) with the local model, and reports agreement
per reference sentence (3-way, lenient = covered+partial vs missed, strict = covered vs the
rest; raw agreement and Cohen's kappa) and the recall each judge gives on the same notes.

Only the local model is called ($0). Its verdicts are cached next to Sonnet's under
model-tagged names (judge.j<N>-qwen2.5-7b.<id>.json); the Sonnet files are only read.
Prompts were tuned on the dev meetings only; the test rows are the held-out check.

Usage: .venv/Scripts/python scripts/step4_calibrate_judge.py [--versions 1 2] [--split dev]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ami_summaries import human_summary
from evaluation.insights_eval import JUDGE_VERSION, judge, judge_filename
from evaluation.judge_agreement import SECTIONS, agreement, recall, statuses
from evaluation.local_judge import LOCAL_JUDGE_VERSION, LOCAL_JUDGES
from pipeline import ami, llm
from pipeline.llm_local import LocalLLMError

EVAL = ami.ROOT / "artifacts" / "eval"
OUT = EVAL / "step4"
DEV = ("IS1008d", "ES2011b", "TS3004c")   # the meetings Step 1 tuned its prompts on
SONNET_PREFIX = f"judge.j{JUDGE_VERSION}."


def find_pairs(root: Path = EVAL) -> list[dict]:
    """Every cached (insights, Sonnet judge) pair: {source, meeting, cache_id, insights, sonnet}.

    asr: judge.j1.<fingerprint>.json pairs with insights.<fingerprint>.json.
    ref: judge.j1.ref-p<P>[.t<T>].json pairs with insights.p<P>[.t<T>].json.
    A judge file whose insights copy was not kept is skipped (nothing to re-judge).
    """
    pairs = []
    for source in ("asr", "ref"):
        for jpath in sorted((root / source).glob(f"*/{SONNET_PREFIX}*.json")):
            cache_id = jpath.name[len(SONNET_PREFIX):-len(".json")]
            stem = cache_id.removeprefix("ref-") if source == "ref" else cache_id
            ipath = jpath.parent / f"insights.{stem}.json"
            if ipath.exists():
                pairs.append({"source": source, "meeting": jpath.parent.name,
                              "cache_id": cache_id, "insights": ipath, "sonnet": jpath})
    return pairs


def local_path(pair: dict, model: str, version: int) -> Path:
    path = pair["sonnet"].parent / judge_filename(pair["cache_id"], model, version)
    if path.name.startswith(SONNET_PREFIX) or path == pair["sonnet"]:
        raise RuntimeError(f"refusing to write over a Sonnet judge file: {path}")
    return path


def compare(rows: list[dict]) -> dict:
    """rows: one per pair with human sentences and both judges' matches -> pooled report."""
    sonnet, local = [], []
    by_sec = {"sonnet": {s: [] for s in SECTIONS}, "local": {s: [] for s in SECTIONS}}
    both_found = shared = 0
    prec = {k: {"n": 0, "sonnet": 0, "local": 0} for k in ("decision", "action")}
    for r in rows:
        ids = [h["id"] for h in r["human"]]
        s, lo = statuses(r["sonnet"], ids), statuses(r["local"], ids)
        sonnet += s
        local += lo
        for h, a, b in zip(r["human"], s, lo, strict=True):
            by_sec["sonnet"][h["section"]].append(a)
            by_sec["local"][h["section"]].append(b)
        s_items = {m["human_id"]: set(m["item_ids"]) for m in r["sonnet"]}
        l_items = {m["human_id"]: set(m["item_ids"]) for m in r["local"]}
        for h, a, b in zip(ids, s, lo, strict=True):
            if a != "missed" and b != "missed":
                both_found += 1
                shared += bool(s_items.get(h, set()) & l_items.get(h, set()))
        for judge_name, matches in (("sonnet", r["sonnet"]), ("local", r["local"])):
            matched = {i for m in matches if m["status"] != "missed" for i in m["item_ids"]}
            for kind, counts in prec.items():
                ours = [it["id"] for it in r["items"] if it["type"] == kind]
                if judge_name == "sonnet":
                    counts["n"] += len(ours)
                counts[judge_name] += sum(i in matched for i in ours)
    return {
        "pairs": len(rows), "sentences": len(sonnet),
        "agreement": agreement(sonnet, local),
        "recall": {"sonnet": recall(by_sec["sonnet"]), "local": recall(by_sec["local"])},
        "precision_lb": {k: {"n": v["n"],
                             "sonnet": round(v["sonnet"] / v["n"], 4) if v["n"] else None,
                             "local": round(v["local"] / v["n"], 4) if v["n"] else None}
                         for k, v in prec.items()},
        "shared_item_rate": {"both_found": both_found,
                             "value": round(shared / both_found, 4) if both_found else None},
    }


def _pct(x) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def print_table(version: int, split: str, rep: dict) -> None:
    ag = rep["agreement"]
    k = {v: ag[v]["kappa"] for v in ("3way", "lenient", "strict")}
    print(f"\nlocal judge v{version} | {split}: {rep['pairs']} note sets, "
          f"{rep['sentences']} reference sentences")
    print(f"  agreement 3-way {_pct(ag['3way']['agreement'])}%  kappa "
          f"3-way {k['3way'] if k['3way'] is not None else '-'}  "
          f"lenient {k['lenient'] if k['lenient'] is not None else '-'} "
          f"(agree {_pct(ag['lenient']['agreement'])}%)  "
          f"strict {k['strict'] if k['strict'] is not None else '-'} "
          f"(agree {_pct(ag['strict']['agreement'])}%)")
    print("  recall          n   Sonnet strict/lenient   local strict/lenient")
    for sec in SECTIONS:
        s, lo = rep["recall"]["sonnet"][sec], rep["recall"]["local"][sec]
        print(f"  {sec:<10} {s['n']:>5}   {_pct(s['strict'])} / {_pct(s['lenient'])}"
              f"          {_pct(lo['strict'])} / {_pct(lo['lenient'])}")
    c = ag["confusion"]
    print("  confusion (rows Sonnet, cols local: covered/partial/missed): "
          + "  ".join(f"{r[:3]} {c[r]['covered']}/{c[r]['partial']}/{c[r]['missed']}"
                      for r in c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, nargs="+", default=[LOCAL_JUDGE_VERSION])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--model", default=llm.LOCAL_MODEL)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run note sets whose local judge call failed before")
    a = ap.parse_args()
    if not llm.is_local(a.model):
        raise SystemExit(f"{a.model} is not a local model: this script never calls Claude")

    llm.RUN_TAG = "step4-judge-calibration"
    pairs = [p for p in find_pairs()
             if a.split == "all" or (p["meeting"] in DEV) == (a.split == "dev")]
    OUT.mkdir(parents=True, exist_ok=True)
    timings_path = OUT / "judge_timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    out_path = OUT / "judge_calibration.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    report.update({"model": a.model, "sonnet_judge": f"{llm.CLAUDE_VERIFIER} (judge v1)",
                   "dev_meetings": list(DEV)})
    report.setdefault("versions", {})
    for version in a.versions:
        rows, failed = [], []
        for p in pairs:
            insights = json.loads(p["insights"].read_text())
            human = human_summary(p["meeting"])
            path = local_path(p, a.model, version)
            key = str(path.relative_to(EVAL)).replace("\\", "/")
            if "error" in timings.get(key, {}) and not a.retry_failed:
                failed.append({"local_file": key, "error": timings[key]["error"]})
                continue
            if not path.exists():
                t0 = time.perf_counter()
                try:
                    matches = judge(p["meeting"], insights, human, llm.RUN_TAG, a.model,
                                    version)
                except LocalLLMError as e:   # e.g. a degenerate reply that never ends
                    timings[key] = {"error": str(e)[:300],
                                    "seconds": round(time.perf_counter() - t0, 1)}
                    timings_path.write_text(json.dumps(timings, indent=1))
                    failed.append({"local_file": key, "error": timings[key]["error"]})
                    print(f"  {key}: FAILED ({timings[key]['error'][:120]})", flush=True)
                    continue
                timings[key] = {"seconds": round(time.perf_counter() - t0, 1),
                                "sentences": len(human)}
                path.write_text(json.dumps(matches, indent=1))
                timings_path.write_text(json.dumps(timings, indent=1))
                print(f"  {p['source']}/{p['meeting']} {p['cache_id']}: {len(human)} sentences "
                      f"in {timings[key]['seconds']} s", flush=True)
            items = [it for it in insights["items"] if it["verification"]["status"] != "rejected"]
            rows.append({"meeting": p["meeting"], "source": p["source"],
                         "cache_id": p["cache_id"], "human": human, "items": items,
                         "sonnet": json.loads(p["sonnet"].read_text()),
                         "local": json.loads(path.read_text()), "key": key})

        vrep = {"settings": LOCAL_JUDGES[version], "failed": failed}
        if failed:
            print(f"\nv{version}: {len(failed)} note set(s) failed and are left out below")
        for split, sel in (("dev", [r for r in rows if r["meeting"] in DEV]),
                           ("test", [r for r in rows if r["meeting"] not in DEV]),
                           ("all", rows)):
            if not sel:
                continue
            vrep[split] = compare(sel)
            secs = [timings[r["key"]]["seconds"] for r in sel if r["key"] in timings]
            vrep[split]["seconds_per_note_set"] = (round(sum(secs) / len(secs), 1)
                                                   if secs else None)
            print_table(version, split, vrep[split])
            print(f"  time per meeting (one note set): {vrep[split]['seconds_per_note_set']} s")
        vrep["per_pair"] = [{"source": r["source"], "meeting": r["meeting"],
                             "cache_id": r["cache_id"], "local_file": r["key"],
                             "seconds": timings.get(r["key"], {}).get("seconds"),
                             "agreement": agreement(statuses(r["sonnet"],
                                                             [h["id"] for h in r["human"]]),
                                                    statuses(r["local"],
                                                             [h["id"] for h in r["human"]]))}
                            for r in rows]
        report["versions"][str(version)] = vrep
    report["current_local_version"] = LOCAL_JUDGE_VERSION
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
