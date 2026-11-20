"""Ask questions about one meeting. Answers come only from the meeting and cite its lines.

Claude (NOTES_LLM=claude): Sonnet reads the whole transcript plus the notes on every question.
  An hour is ~15k tokens; the transcript block is prompt-cached, so follow-up questions within
  5 minutes read it at a tenth of the price (about $0.03 for the first question, under $0.01
  after).
Local (NOTES_LLM=local): a 7B model can't hold a long meeting, so the question first picks the
  best-matching stretches of transcript by keyword (BM25), and the model reads only those.

Citations are checked here: IDs that don't exist (or, locally, weren't shown to the model)
are dropped, so every play button in an answer points at a real line.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from evaluation.local_judge import bm25_rank
from pipeline import llm
from pipeline.insights.extract import format_transcript
from pipeline.insights.prompts import TRANSCRIPT_NOTE
from pipeline.insights.verify import render

PROMPT_VERSION = 2
WINDOW, STEP = 10, 5      # local retrieval: 10-utterance stretches, overlapping by half
LOCAL_WINDOWS = 8         # stretches the local model reads per question
HISTORY_TURNS = 6         # earlier questions kept for follow-ups ("and who owns that?")

SYSTEM = f"""You answer questions about one meeting, using only that meeting's transcript and \
notes, which are given below.
{TRANSCRIPT_NOTE}

Rules:
- Answer only from the transcript. Never guess or add outside knowledge. If the meeting does \
not cover it, set found to false and say briefly that it wasn't discussed (mention the closest \
thing that was, if any).
- Be direct: lead with the answer, then the needed detail. Quote numbers exactly as said.
- Split the answer into parts of one or two sentences. Each part cites the utterance IDs that \
support it (as few as needed). A part that only frames the answer may cite nothing.
- Refer to people by their speaker label alone (e.g. SPEAKER_01), exactly as in the \
transcript, never followed by their name: the app replaces each label with the name. Names \
known so far are listed under Speakers.
- The notes are a machine summary: prefer the transcript when they disagree."""


class AnswerPart(BaseModel):
    text: str
    utterance_ids: list[str] = Field(description='Supporting utterance IDs, e.g. ["u0142"]')


class Answer(BaseModel):
    found: bool = Field(description="false if the meeting does not cover the question")
    parts: list[AnswerPart]


class LocalAnswerPart(AnswerPart):
    text: str = Field(max_length=600)
    utterance_ids: list[str] = Field(max_length=6, description='Supporting utterance IDs, e.g. ["u0142"]')


class LocalAnswer(Answer):
    parts: list[LocalAnswerPart] = Field(max_length=6)


def _utterances(rows: list[dict]) -> list[dict]:
    """DB transcript rows -> the shape the prompts use (IDs u0000, u0001, ...)."""
    return [{"id": f"u{r['seq']:04d}", "start": r["start_ms"] / 1000, "speaker": r["speaker_label"],
             "text": r["text"]} for r in rows]


def _speakers(speakers: list[dict]) -> str:
    named = [f"{s['cluster_label']} = {s['display_name']}" for s in speakers if s.get("display_name")]
    return "Speakers: " + ("; ".join(named) if named else "no names known yet")


def _notes(payload: dict | None) -> str:
    if not payload:
        return "Notes: none yet."
    kept = [it for it in payload["items"] if it["verification"]["status"] != "rejected"]
    parts = [f"Notes (summary): {payload['summary']}"]
    for t in payload["topics"]:
        lines = [f"- {render(it)}" for it in kept if it.get("topic_id") == t["id"]]
        parts.append(f"## {t['title']} ({t['start_id']}-{t['end_id']})\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _retrieve(utts: list[dict], query: str) -> list[dict]:
    """The best-matching stretches of the meeting, merged and back in time order."""
    starts = list(range(0, max(1, len(utts) - WINDOW + STEP), STEP))
    docs = [" ".join(f"{u['speaker']} {u['text']}" for u in utts[a:a + WINDOW]) for a in starts]
    keep = sorted(bm25_rank(query, docs)[:LOCAL_WINDOWS])
    seen: set[int] = set()
    for w in keep:
        seen.update(range(starts[w], min(len(utts), starts[w] + WINDOW)))
    return [utts[i] for i in sorted(seen)]


def _excerpts(utts: list[dict], picked: list[dict], title: str) -> str:
    """Picked lines, with '...' where the transcript skips ahead."""
    pos = {u["id"]: i for i, u in enumerate(utts)}
    out, last = [f"Meeting: {title}", "Excerpts that best match the question:"], None
    for u in picked:
        if last is not None and pos[u["id"]] != last + 1:
            out.append("...")
        s = int(u["start"])
        out.append(f"[{u['id']} | {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d} | "
                   f"{u['speaker']}] {u['text']}")
        last = pos[u["id"]]
    return "\n".join(out)


def _label_only(text: str, speakers: list[dict]) -> str:
    """'SPEAKER_03 (Jeanne)' -> 'SPEAKER_03'. The app swaps labels for names, so a name the model
    added beside the label would show twice ("Jeanne (Jeanne)")."""
    names = {s["cluster_label"]: s["display_name"] for s in speakers if s.get("display_name")}

    def one(m: re.Match) -> str:
        label, name = m.group(1), m.group(2).strip()
        return label if name.casefold() == names.get(label, "").casefold() else m.group(0)
    return re.sub(r"(SPEAKER_\d{2}) \(([^()]{1,40})\)", one, text)


def ask(meeting_id: str, title: str, transcript_rows: list[dict], speakers: list[dict],
        notes: dict | None, question: str, history: list[dict] | None = None) -> dict:
    """Answer one question. `history` holds earlier turns: [{"question", "answer"}]."""
    utts = _utterances(transcript_rows)
    history = (history or [])[-HISTORY_TURNS:]
    turns = []
    for h in history:
        turns += [{"role": "user", "content": h["question"]},
                  {"role": "assistant", "content": h["answer"]}]

    if llm.PROVIDER == "claude":
        model, schema = llm.CLAUDE_VERIFIER, Answer
        context = "\n\n".join([format_transcript(utts, title, None), _speakers(speakers), _notes(notes)])
        system = [{"type": "text", "text": SYSTEM},
                  {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}}]
        allowed = {u["id"] for u in utts}
        messages = turns + [{"role": "user", "content": question}]
    else:
        model, schema = llm.LOCAL_MODEL, LocalAnswer
        query = " ".join([h["question"] for h in history[-1:]] + [question])
        picked = _retrieve(utts, query)
        allowed = {u["id"] for u in picked}
        summary = f"Notes (summary): {notes['summary']}" if notes else "Notes: none yet."
        context = "\n\n".join([_excerpts(utts, picked, title), _speakers(speakers), summary])
        system = SYSTEM + "\n\nOnly part of the meeting is shown: if the excerpts don't cover " \
                          "the question, say so rather than guess."
        # a 7B model follows a question best when the context sits right next to it
        messages = turns + [{"role": "user", "content": context + "\n\nQuestion: " + question}]

    ans, rec = llm.parse(schema, model=model, system=system, messages=messages, purpose="ask",
                         meeting=meeting_id, effort="medium", max_tokens=2000)
    parts = [{"text": _label_only(p.text, speakers), "utterance_ids": [i for i in p.utterance_ids if i in allowed]}
             for p in ans.parts if p.text.strip()]
    return {"found": ans.found, "parts": parts, "model": model,
            "cost_usd": rec.get("cost_usd", 0.0), "prompt_version": PROMPT_VERSION}
