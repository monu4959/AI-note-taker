"""Step 3: how often are transcript-cue names right? Measured on 6 AMI series (24 meetings).

Ground truth comes from the HUMAN transcript, independent of our pipeline: when AMI speaker
FIE088 says "I'm Christine" / "my name is Christine", FIE088 is Christine for the whole
series (AMI global speaker IDs are stable across a series).

For each meeting, name_cues runs on OUR transcript (Whisper + pyannote), and each accepted
name lands on a diarization cluster. The cluster's true speaker is the AMI speaker covering
most of its speech. A name is
  correct    the true speaker's introduced name matches it (fuzzy: ASR spells names freely)
  wrong      the true speaker introduced themselves with a different name, or the name
             belongs to another speaker
  unchecked  the true speaker never introduced themselves in the human transcript
LLM calls are cached in artifacts/eval/step3/cues/.

Usage: .venv/Scripts/python scripts/step3_eval_cues.py
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import ami, llm, name_cues
from pipeline.run import MEETINGS
from scripts.step3_calibrate import SERIES, truth

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "eval" / "step3" / "cues"
INTRO = re.compile(r"\b(?:I'm|I am|my name is|my name's|this is)\s+([A-Z][a-z]{2,})")
NOT_NAMES = {"The", "Going", "Not", "Just", "Gonna", "Sure", "Very", "Really", "Actually",
             "Also", "Still", "Here", "Sorry", "Done", "Okay", "Fine", "Glad", "Happy", "Our",
             "Your", "Project", "Marketing", "User", "Industrial", "Interface"}


def intro_names(meeting: str) -> dict[str, set[str]]:
    """Global speaker ID -> names they introduce themselves with (human transcript)."""
    out: dict[str, set[str]] = {}
    for seg in ami.reference_segments(meeting, punctuation=True):
        for m in INTRO.finditer(seg["words"]):
            if m.group(1) not in NOT_NAMES:
                out.setdefault(seg["speaker"], set()).add(m.group(1))
    return out


def same_name(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a == b or a.startswith(b) or b.startswith(a) or \
        difflib.SequenceMatcher(None, a, b).ratio() >= 0.7


def cues_for(meeting: str) -> dict:
    from pipeline import llm

    # Sonnet's cues keep the original name; any other model gets its own file
    tag = "" if llm.VERIFIER == llm.CLAUDE_VERIFIER else "." + llm.stage_suffix(llm.VERIFIER)
    path = OUT / f"{meeting}{tag}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    utts = json.loads((MEETINGS / f"ami-{meeting}" / "assign" / "utterances.json")
                      .read_text(encoding="utf-8"))
    cues, rec = name_cues.propose(utts, "Design team meeting", None, meeting=f"ami-{meeting}")
    res = {"cues": [c.model_dump() for c in cues], **name_cues.resolve(cues, utts),
           "cost_usd": rec["cost_usd"]}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> None:
    llm.RUN_TAG = "step3-cues"
    rows, total_cost = [], 0.0
    for split, series in SERIES.items():
        for s in series:
            names: dict[str, set[str]] = {}
            for x in "abcd":
                for spk, ns in intro_names(f"{s}{x}").items():
                    names.setdefault(spk, set()).update(ns)
            for x in "abcd":
                m = f"{s}{x}"
                res = cues_for(m)
                total_cost += res.get("cost_usd", 0.0)
                t = truth(m, split, MEETINGS / f"ami-{m}" / "diarize")
                for label, a in res["accepted"].items():
                    spk = t.get(label)
                    own = names.get(spk, set())
                    others = {n for k, v in names.items() if k != spk for n in v}
                    if spk and own and any(same_name(a["name"], n) for n in own):
                        verdict = "correct"
                    elif (spk and own) or any(same_name(a["name"], n) for n in others) or spk is None:
                        verdict = "wrong"
                    else:
                        verdict = "unchecked"
                    rows.append({"meeting": m, "label": label, "name": a["name"], "true_speaker": spk,
                                 "true_names": sorted(own), "kinds": a["kinds"], "verdict": verdict})

    (OUT.parent / "cues_eval.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    c = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "wrong", "unchecked")}
    checked = c["correct"] + c["wrong"]
    print(f"{len(rows)} names proposed and accepted across 24 meetings (LLM cost ${total_cost:.2f})")
    print(f"  correct {c['correct']}, wrong {c['wrong']}, unchecked {c['unchecked']}"
          f"  -> precision on checkable names {c['correct'] / checked:.1%}" if checked else "")
    for r in rows:
        if r["verdict"] != "correct":
            print(f"  {r['verdict']:9} {r['meeting']} {r['label']} -> {r['name']!r} "
                  f"(true {r['true_speaker']}, introduced as {r['true_names']}, {r['kinds']})")


if __name__ == "__main__":
    main()
