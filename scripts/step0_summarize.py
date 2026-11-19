"""Step 0.6d: one LLM summary of one meeting, from our own diarized transcript.

Deliberately simple: one call, markdown output, every point cites utterance IDs.
The structured schema, per-topic extraction and verification pass are Step 1.

Usage: .venv/Scripts/python scripts/step0_summarize.py MEETING
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic

import pipeline  # noqa: F401  (loads .env)
from pipeline import ami

MODEL = "claude-opus-5-5"
EFFORT = "high"
UTT_DIR = ami.ROOT / "artifacts" / "step0" / "utterances"
OUT = ami.ROOT / "artifacts" / "step0" / "summaries"

SYSTEM = """You summarize meeting transcripts. The transcript is machine-generated: speaker \
labels are anonymous diarization clusters and some words are misheard.

Write markdown with these sections: Summary (3-5 sentences), Decisions, Action items \
(owner and due date only if someone said them), Metrics and numbers, Open questions.
Rules:
- Only include what was actually said. An empty section is fine: write "None."
- End every bullet with the utterance IDs that support it, e.g. [u0142, u0144].
- Never invent owners, dates or numbers."""


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def transcript_text(meeting: str) -> str:
    utts = json.loads((UTT_DIR / f"{meeting}.json").read_text())
    lines = [f"[{u['id']} | {fmt_ts(u['start'])} | {u['speaker']}] {u['text']}" for u in utts]
    return f"Meeting: {meeting} (AMI corpus)\n\n" + "\n".join(lines)


def main(meeting: str) -> None:
    from pipeline import llm

    if llm.PROVIDER != "claude":   # this Step 0 baseline calls Claude directly
        raise SystemExit("Claude is switched off (NOTES_LLM=local); this script needs it")
    OUT.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    t0 = time.perf_counter()
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        output_config={"effort": EFFORT},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": transcript_text(meeting)}],
    ) as stream:
        msg = stream.get_final_message()
    elapsed = time.perf_counter() - t0

    if msg.stop_reason == "refusal":
        raise SystemExit(f"refused: {msg.stop_details}")
    text = "".join(b.text for b in msg.content if b.type == "text")
    (OUT / f"{meeting}.md").write_text(text, encoding="utf-8")
    meta = {
        "meeting": meeting,
        "model": msg.model,
        "effort": EFFORT,
        "stop_reason": msg.stop_reason,
        "input_tokens": msg.usage.input_tokens,
        "output_tokens": msg.usage.output_tokens,
        "elapsed_s": round(elapsed, 1),
        "request_id": getattr(msg, "_request_id", None),  # absent on beta parsed messages
    }
    (OUT / f"{meeting}.meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main(sys.argv[1])
