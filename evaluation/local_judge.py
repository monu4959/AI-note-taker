"""The notes judge on a local model (Ollama, $0), calibrated against the Sonnet judge.

A 7B model judging every reference sentence against every note in one call is the v1
baseline (same prompt as the Sonnet judge). Later versions split the work:
- a deterministic lexical pre-filter (BM25 over the rendered notes) picks the top-k candidate
  notes for each reference sentence, so the model reads a short list;
- reference sentences are judged in small batches (1 = one call per sentence);
- the model writes its reason before the verdict (the schema's field order).

Every version's settings stay in LOCAL_JUDGES, so an old cached result can be reproduced.
`scripts/step4_calibrate_judge.py` measures each version against the Sonnet verdicts.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from pipeline import llm
from pipeline.insights.verify import render

# version -> settings. batch: reference sentences per call (None = all in one call).
# top_k: candidate notes per sentence from the pre-filter (None = all notes).
LOCAL_JUDGES: dict[int, dict] = {
    1: {"batch": None, "top_k": None, "prompt": "sonnet"},
    2: {"batch": 1, "top_k": 12, "prompt": "focused"},
    3: {"batch": 1, "top_k": 12, "prompt": "elements"},
}
LOCAL_JUDGE_VERSION = 2

FOCUSED_SYSTEM = """You check whether a system's meeting notes convey one sentence from a \
human-written summary of the same meeting.

You get the reference sentence and the system notes most similar to it (ids like d3, a1, k12, \
q2, m4). Decide:
- covered: one or more notes state the same thing. Wording may differ, and it may be split \
across notes. A decision written up as a key point still counts if the same outcome is stated.
- partial: the notes convey part of it but miss or contradict an important element \
(who, what exactly was decided, a condition or a number).
- missed: no note conveys it.
For a "problems" sentence (an issue, difficulty or open question), notes that raise or \
discuss the same issue cover it, even if they phrase it as a question or a key point.
Judge meaning, not wording. Speaker labels in the notes are anonymous (e.g. SPEAKER_02), \
while the reference uses roles (e.g. "the Marketing Expert"): never count that as a \
difference.
First list the note ids you relied on (empty if none), then a one-sentence reason, then the \
status."""

ELEMENTS_SYSTEM = """You check whether a system's meeting notes convey one sentence from a \
human-written summary of the same meeting.

You get the reference sentence and the system notes most similar to it (ids like d3, a1, k12, \
q2, m4). Split the reference sentence into its key elements (usually 1 to 3: what was \
decided / to be done / the issue, plus who, a condition or a number when the sentence has \
one). For each element give the ids of the notes that state it and a verdict:
- stated: the notes state this element (wording may differ; a decision written as a key \
point counts; for an issue or problem, notes that raise or discuss it count).
- contradicted: the notes state something different for this element.
- absent: no note states it.
Judge meaning, not wording. Speaker labels in the notes are anonymous (e.g. SPEAKER_02), \
while the reference uses roles (e.g. "the Marketing Expert"): a role is never an element \
to check against a speaker label."""

_STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be
been being it its this that these those there their they them he she his her we our you your i
me my not no so than then too very can will would should could may might must do does did have
has had which who whom what when where why how all any each also only about into over under
up down out off just more most some such own same other team meeting group""".split())  # noqa: SIM905


# Length caps go into the JSON schema, so constrained decoding cannot loop forever on a list
# of ids (v1, without caps, once wrote ids until it hit max_tokens).
IdList = Field(max_length=8)


class LocalMatch(BaseModel):
    human_id: str
    item_ids: list[str] = IdList
    reason: str = Field(max_length=400)
    status: Literal["covered", "partial", "missed"]


class LocalMatching(BaseModel):
    matches: list[LocalMatch]


class Element(BaseModel):
    element: str = Field(max_length=200)
    note_ids: list[str] = IdList
    verdict: Literal["stated", "contradicted", "absent"]


class ElementVerdict(BaseModel):
    elements: list[Element] = Field(max_length=5)


def status_from_elements(verdicts: list[str]) -> str:
    """All elements stated -> covered; none stated -> missed; otherwise partial."""
    stated = sum(v == "stated" for v in verdicts)
    if verdicts and stated == len(verdicts):
        return "covered"
    return "partial" if stated else "missed"


class LocalVerdict(BaseModel):
    item_ids: list[str] = IdList
    reason: str = Field(max_length=400)
    status: Literal["covered", "partial", "missed"]


def tokens(text: str) -> list[str]:
    """Lower-case content words with a light suffix strip (buttons -> button, designed ->
    design), so a sentence and a note phrased a little differently still share terms."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def bm25_rank(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[int]:
    """Indices of docs, best match first (ties keep document order)."""
    toks = [tokens(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avg = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = tokens(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def candidates(sentence: str, items: list[dict], top_k: int | None) -> list[dict]:
    """The top_k notes most similar to the sentence, in the notes' original order."""
    if top_k is None or top_k >= len(items):
        return list(items)
    keep = set(bm25_rank(sentence, [render(it) for it in items])[:top_k])
    return [it for i, it in enumerate(items) if i in keep]


def batches(seq: list, size: int | None) -> list[list]:
    if not seq:
        return []
    size = size or len(seq)
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _clean(human_id: str, status: str, item_ids: list[str], reason: str,
           shown: set[str]) -> dict:
    ids = [] if status == "missed" else list(dict.fromkeys(i for i in item_ids if i in shown))
    return {"human_id": human_id, "status": status, "item_ids": ids, "reason": reason}


def judge_local(meeting: str, insights: dict, human: list[dict], tag: str, model: str,
                version: int = LOCAL_JUDGE_VERSION) -> list[dict]:
    """Same output as insights_eval.judge: one {human_id, status, item_ids, reason} per
    reference sentence."""
    if not llm.is_local(model):
        raise ValueError(f"{model} is not a local model")
    cfg = LOCAL_JUDGES[version]
    items = [it for it in insights["items"] if it["verification"]["status"] != "rejected"]
    if cfg["prompt"] == "sonnet":
        return _judge_all(meeting, items, human, tag, model, cfg["batch"])

    if cfg["prompt"] == "elements" and cfg["batch"] != 1:
        raise ValueError("the elements prompt judges one sentence per call")
    out = []
    for group in batches(human, cfg["batch"]):
        shown_items: dict[str, dict] = {}
        for h in group:
            for it in candidates(h["text"], items, cfg["top_k"]):
                shown_items[it["id"]] = it
        order = [it for it in items if it["id"] in shown_items]
        notes = "\n".join(f"{it['id']}: {render(it)}" for it in order)
        shown = set(shown_items)
        if cfg["prompt"] == "elements":
            h = group[0]
            v, _ = llm.parse(
                ElementVerdict, model=model, system=ELEMENTS_SYSTEM,
                messages=[{"role": "user", "content":
                           f"REFERENCE SENTENCE [{h['section']}]\n{h['text']}\n\n"
                           f"SYSTEM NOTES\n{notes}"}],
                purpose=f"judge-local:{tag}", meeting=meeting, max_tokens=1000)
            status =status_from_elements([e.verdict for e in v.elements])
            ids = [i for e in v.elements if e.verdict == "stated" for i in e.note_ids]
            reason = "; ".join(f"{e.element}: {e.verdict}" for e in v.elements)
            out.append(_clean(h["id"], status, ids, reason, shown))
            continue
        if len(group) == 1:
            h = group[0]
            v, _ = llm.parse(
                LocalVerdict, model=model, system=FOCUSED_SYSTEM,
                messages=[{"role": "user", "content":
                           f"REFERENCE SENTENCE [{h['section']}]\n{h['text']}\n\n"
                           f"SYSTEM NOTES\n{notes}"}],
                purpose=f"judge-local:{tag}", meeting=meeting, max_tokens=1000)
            out.append(_clean(h["id"], v.status, v.item_ids, v.reason, shown))
            continue
        refs = "\n".join(f"{h['id']} [{h['section']}]: {h['text']}" for h in group)
        res, _ = llm.parse(
            LocalMatching, model=model, system=FOCUSED_SYSTEM,
            messages=[{"role": "user", "content":
                       f"SYSTEM NOTES\n{notes}\n\nREFERENCE SENTENCES\n{refs}\n\n"
                       "Return one match per reference sentence."}],
            purpose=f"judge-local:{tag}", meeting=meeting, max_tokens=600 * len(group))
        got = {m.human_id: m for m in res.matches}
        for h in group:
            m = got.get(h["id"])
            out.append(_clean(h["id"], m.status, m.item_ids, m.reason, shown) if m else
                       _clean(h["id"], "missed", [], "no verdict returned", shown))
    return out


def _judge_all(meeting: str, items: list[dict], human: list[dict], tag: str, model: str,
               batch: int | None) -> list[dict]:
    """v1: the Sonnet judge's prompt and schema, unchanged, on the local model."""
    from evaluation.insights_eval import JUDGE_SYSTEM, Matching

    notes = "\n".join(f"{it['id']}: {render(it)}" for it in items)
    shown = {it["id"] for it in items}
    out = []
    for group in batches(human, batch):
        refs = "\n".join(f"{h['id']} [{h['section']}]: {h['text']}" for h in group)
        res, _ = llm.parse(
            Matching, model=model, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content":
                       f"SYSTEM NOTES\n{notes}\n\nREFERENCE SENTENCES\n{refs}\n\n"
                       "Return one match per reference sentence."}],
            purpose=f"judge-local:{tag}", meeting=meeting, max_tokens=8000)
        got = {m.human_id: m for m in res.matches}
        for h in group:
            m = got.get(h["id"])
            out.append(_clean(h["id"], m.status, m.item_ids, m.reason, shown) if m else
                       _clean(h["id"], "missed", [], "no verdict returned", shown))
    return out
