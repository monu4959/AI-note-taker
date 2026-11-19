"""Step 3 gate: calibrate the voiceprint threshold on AMI scenario series (same 4 people in
meetings a-d). Enroll everyone from meeting a, identify them in b-d, sweep the threshold.

Three settings:
  attendees  candidates = the series' 4 people (the attendee list is known)
  memory     candidates = everyone enrolled from all series (24 people)
  stranger   one person per series is never enrolled; their clusters must stay unnamed

Truth for a cluster = the AMI speaker covering most of its speech, if >= 50% (else "mixed":
any name on it counts as wrong). No LLM calls.

Usage: .venv/Scripts/python scripts/step3_calibrate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import ami, speaker_id
from pipeline.diarization import list_to_annotation
from pipeline.run import MEETINGS, PipelineConfig, run

SERIES = {"test": ["IS1009", "ES2004", "TS3003"], "dev": ["IS1008", "ES2011", "TS3004"]}
THRESHOLDS = [round(t, 3) for t in np.arange(0.30, 0.901, 0.025)]
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "eval" / "step3"


def ensure_processed(meeting: str) -> Path:
    d = MEETINGS / f"ami-{meeting}" / "diarize"
    if not (d / "embeddings.npy").exists():
        print(f"processing {meeting} (GPU stages)...", flush=True)
        run(f"ami-{meeting}", ami.audio_path(meeting), PipelineConfig(insights=False, name_cues=False))
    return d


def truth(meeting: str, split: str, diarize_dir: Path) -> dict[str, str | None]:
    """cluster label -> AMI global speaker ID (or None when no speaker has >= 50%)."""
    ref = ami.reference_rttm(meeting, split)
    turns = json.loads((diarize_dir / "diarization.json").read_text())["exclusive"]
    hyp = list_to_annotation(turns, uri=meeting)
    out = {}
    for label in hyp.labels():
        tl = hyp.label_timeline(label)
        total = tl.duration()
        best, best_s = None, 0.0
        for spk in ref.labels():
            s = tl.crop(ref.label_timeline(spk), mode="intersection").duration()
            if s > best_s:
                best, best_s = spk, s
        out[label] = best if total and best_s / total >= 0.5 else None
    return out


def load() -> list[dict]:
    """One record per meeting: series, letter, clusters, truth."""
    recs = []
    for split, series in SERIES.items():
        for s in series:
            for x in "abcd":
                m = f"{s}{x}"
                d = ensure_processed(m)
                recs.append({"meeting": m, "series": s, "letter": x,
                             "clusters": speaker_id.clusters_from_diarization(d),
                             "truth": truth(m, split, d)})
    return recs


def enroll(recs: list[dict]) -> dict[str, list[np.ndarray]]:
    """From each series' meeting a: each person's largest cluster that is mostly them."""
    gallery: dict[str, list[np.ndarray]] = {}
    for r in recs:
        if r["letter"] != "a":
            continue
        best: dict[str, speaker_id.Cluster] = {}
        for c in r["clusters"]:
            p = r["truth"].get(c.label)
            if p and (p not in best or c.speech_s > best[p].speech_s):
                best[p] = c
        for p, c in best.items():
            gallery.setdefault(p, []).append(c.embedding)
    return gallery


def evaluate(recs, gallery, threshold, setting) -> dict:
    named = correct = wrong_known = false_stranger = eligible = strangers = 0
    for r in recs:
        if r["letter"] == "a":
            continue
        people = {p for p in r["truth"].values() if p}
        runs = [None] if setting != "stranger" else sorted(people)
        for held_out in runs:
            g = {p: v for p, v in gallery.items() if p != held_out}
            cands = people if setting == "attendees" else None
            matches = {m.label: m for m in speaker_id.identify(r["clusters"], g, threshold,
                                                               candidates=cands)}
            for c in r["clusters"]:
                t = r["truth"].get(c.label)
                if c.speech_s < speaker_id.MIN_SPEECH_S:
                    continue
                is_stranger = t is not None and t == held_out
                if setting == "stranger" and not is_stranger:
                    continue
                if is_stranger:
                    strangers += 1
                elif t is not None:
                    eligible += 1
                m = matches.get(c.label)
                if not m:
                    continue
                named += 1
                if is_stranger:
                    false_stranger += 1
                elif m.person == t:
                    correct += 1
                else:
                    wrong_known += 1
    return {"setting": setting, "threshold": threshold, "named": named, "correct": correct,
            "wrong": wrong_known + false_stranger, "eligible": eligible,
            "strangers": strangers, "false_stranger": false_stranger,
            "precision": correct / named if named else None,
            "coverage": correct / eligible if eligible else None,
            "stranger_named_rate": false_stranger / strangers if strangers else None}


def main() -> None:
    recs = load()
    gallery = enroll(recs)
    n_test = sum(1 for r in recs if r["letter"] != "a"
                 for c in r["clusters"] if c.speech_s >= speaker_id.MIN_SPEECH_S)
    mixed = sum(1 for r in recs if r["letter"] != "a" for c in r["clusters"]
                if c.speech_s >= speaker_id.MIN_SPEECH_S and r["truth"].get(c.label) is None)
    print(f"{len(recs)} meetings, {len(gallery)} people enrolled, "
          f"{n_test} test clusters with >= {speaker_id.MIN_SPEECH_S:.0f} s speech ({mixed} mixed)")

    rows = [evaluate(recs, gallery, t, s) for s in ("attendees", "memory", "stranger")
            for t in THRESHOLDS]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(rows, indent=1))

    def fmt(v):
        return "  -  " if v is None else f"{v:5.1%}"

    print(f"{'thr':>5} | {'attendees prec/cov':>19} | {'memory prec/cov':>17} | stranger named")
    for t in THRESHOLDS:
        a, m, s = (next(r for r in rows if r["setting"] == k and r["threshold"] == t)
                   for k in ("attendees", "memory", "stranger"))
        print(f"{t:5.3f} | {fmt(a['precision'])} {fmt(a['coverage'])} ({a['named']:2d}) | "
              f"{fmt(m['precision'])} {fmt(m['coverage'])} ({m['named']:2d}) | "
              f"{fmt(s['stranger_named_rate'])} ({s['false_stranger']}/{s['strangers']})")

    # similarity distributions: same person vs different person (for the report)
    same, diff = [], []
    for r in recs:
        if r["letter"] == "a":
            continue
        people, S = speaker_id.score_matrix(r["clusters"], gallery)
        for i, c in enumerate(r["clusters"]):
            t = r["truth"].get(c.label)
            if t is None or c.speech_s < speaker_id.MIN_SPEECH_S:
                continue
            for j, p in enumerate(people):
                (same if p == t else diff).append(S[i, j])
    print(f"same-person similarity: median {np.median(same):.3f}, 5th pct {np.percentile(same, 5):.3f}"
          f" (n={len(same)})")
    print(f"other-person similarity: median {np.median(diff):.3f}, 99th pct "
          f"{np.percentile(diff, 99):.3f}, max {np.max(diff):.3f} (n={len(diff)})")


if __name__ == "__main__":
    main()
