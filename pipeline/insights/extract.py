"""Insight extraction.

per_topic (default): segment -> extract per topic (full transcript cached, topic span varies)
                     -> verify -> summarize from verified items only.
single_pass:         one call returns everything -> verify (summary kept as written).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from pipeline import llm, llm_local
from pipeline.insights import prompts
from pipeline.insights.schema import (
    SCHEMA_VERSION,
    ActionItem,
    Item,
    MeetingInsights,
    Metric,
    Segmentation,
    Summary,
    TopicExtraction,
)
from pipeline.insights.verify import verify

log = logging.getLogger("insights")

ITEM_LISTS = {"decisions": "decision", "action_items": "action", "metrics": "metric",
              "open_questions": "open_question", "qa": "qa"}
ID_PREFIX = {"decision": "d", "action": "a", "metric": "m", "open_question": "q", "key_point": "k",
             "qa": "qa"}


def _ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def format_transcript(utterances: list[dict], title: str, attendees: list[str] | None,
                      max_chars: int | None = None) -> str:
    header = f"Meeting: {title}"
    if attendees:
        header += "\nAttendees: " + ", ".join(attendees)
    def text(t: str) -> str:
        return t if max_chars is None or len(t) <= max_chars else t[:max_chars] + "..."
    lines = [f"[{u['id']} | {_ts(u['start'])} | {u['speaker']}] {text(u['text'])}"
             for u in utterances]
    return header + "\n\n" + "\n".join(lines)


def _fix_spans(spans, ids: list[str]) -> list[dict]:
    """Make topic spans valid: known IDs, sorted, contiguous, covering the whole meeting."""
    if not ids:
        return []
    pos = {u: i for i, u in enumerate(ids)}
    clean = sorted(({"title": s.title, "a": pos[s.start_id], "b": pos[s.end_id]}
                    for s in spans if s.start_id in pos and s.end_id in pos
                    and pos[s.start_id] <= pos[s.end_id]), key=lambda s: s["a"])
    if not clean:
        return [{"title": "Meeting", "start_id": ids[0], "end_id": ids[-1]}]
    clean[0]["a"] = 0
    for prev, nxt in zip(clean, clean[1:]):
        nxt["a"] = max(nxt["a"], prev["a"] + 1)
        prev["b"] = nxt["a"] - 1  # close gaps and remove overlaps
    clean = [s for s in clean if s["a"] < len(ids)]
    clean[-1]["b"] = len(ids) - 1
    return [{"title": s["title"], "start_id": ids[s["a"]], "end_id": ids[s["b"]]}
            for s in clean if s["a"] <= s["b"]]


def _flatten(ext, topic_id: str | None = None) -> list[dict]:
    items = []
    for field, kind in ITEM_LISTS.items():
        for obj in getattr(ext, field, []):
            items.append({"type": kind, "topic_id": topic_id, **obj.model_dump()})
    for kp in getattr(ext, "key_points", []):
        items.append({"type": "key_point", "topic_id": topic_id, **kp.model_dump()})
    return items


def _number(items: list[dict]) -> None:
    counters: dict[str, int] = {}
    for it in items:
        p = ID_PREFIX[it["type"]]
        counters[p] = counters.get(p, 0) + 1
        it["id"] = f"{p}{counters[p]}"


LOCAL_CONTEXT = 30   # utterances of context either side of a topic (local model)


# Local schemas: same fields as the Claude ones (so _flatten and verify work unchanged), with
# length caps. Ollama turns the schema into a grammar, so the caps hold during generation:
# without them qwen2.5:7b sometimes loops ("u0421, u0422, u0423, ...") until max_tokens.
class LocalEvidence(BaseModel):
    utterance_ids: list[str] = Field(max_length=6, description='Supporting utterance IDs, e.g. ["u0142", "u0144"]')


class LocalItem(Item):
    text: str = Field(max_length=400)
    evidence: LocalEvidence


class LocalActionItem(ActionItem):
    task: str = Field(max_length=300)
    evidence: LocalEvidence


class LocalMetric(Metric):
    name: str = Field(max_length=120, description='What is measured, e.g. "Q3 churn"')
    value: str = Field(max_length=40, description='As said, e.g. "4.2"')
    unit: str | None = Field(max_length=30, description='e.g. "%", "euros"; null if not said')
    context: str | None = Field(max_length=200, description='e.g. "down from 5.1"')
    evidence: LocalEvidence


class LocalCommitments(BaseModel):
    decisions: list[LocalItem] = Field(max_length=10)
    action_items: list[LocalActionItem] = Field(max_length=10)


class LocalFacts(BaseModel):
    key_points: list[LocalItem] = Field(max_length=3)
    metrics: list[LocalMetric] = Field(max_length=10)
    open_questions: list[LocalItem] = Field(max_length=5)


def _window_text(utterances: list[dict], a: int, b: int, title: str,
                 attendees: list[str] | None) -> str:
    """Utterances [a, b) to take notes from, marked >>>, with read-only context around."""
    lo = max(0, a - prompts.LOCAL_WINDOW_CONTEXT)
    hi = min(len(utterances), b + prompts.LOCAL_WINDOW_CONTEXT)
    header = f"Meeting: {title}" + (("\nAttendees: " + ", ".join(attendees)) if attendees else "")
    lines = [f"{'>>> ' if a <= n < b else '    '}[{u['id']} | {_ts(u['start'])} | {u['speaker']}] "
             f"{u['text']}" for n, u in enumerate(utterances[lo:hi], start=lo)]
    return header + "\n\n" + "\n".join(lines)


def extract_local_windows(utterances, topics, meeting, title, attendees, model,
                          skipped: list[dict] | None = None) -> list[dict]:
    """Local model: per topic, windows of LOCAL_WINDOW utterances, two focused passes each.
    A window whose reply never validates is skipped and listed in `skipped`."""
    skipped = skipped if skipped is not None else []
    pos = {u["id"]: n for n, u in enumerate(utterances)}
    items: list[dict] = []
    for t in topics:
        a0, b0 = pos[t["start_id"]], pos[t["end_id"]] + 1
        for a in range(a0, b0, prompts.LOCAL_WINDOW):
            b = min(b0, a + prompts.LOCAL_WINDOW)
            text = _window_text(utterances, a, b, title, attendees)
            ask = f"Topic: {t['title']}. Take notes from the >>> lines ({utterances[a]['id']} to {utterances[b - 1]['id']})."
            allowed = {u["id"] for u in utterances[a:b]}
            for schema, system, purpose in ((LocalCommitments, prompts.LOCAL_COMMITMENTS_SYSTEM, "extract:commitments"),
                                            (LocalFacts, prompts.LOCAL_FACTS_SYSTEM, "extract:facts")):
                try:
                    ext, _ = llm.parse(schema, model=model, system=system,
                                       messages=[{"role": "user", "content": text + "\n\n" + ask}],
                                       purpose=purpose, meeting=meeting, max_tokens=3000)
                except llm_local.LocalLLMInvalidOutput as e:   # a bad reply: skip this window
                    log.warning("%s %s-%s skipped: %s", purpose, utterances[a]["id"],
                                utterances[b - 1]["id"], e)
                    skipped.append({"purpose": purpose, "from": utterances[a]["id"],
                                    "to": utterances[b - 1]["id"], "error": str(e)[:300]})
                    continue
                for it in _flatten(ext, t["id"]):
                    # keep citations inside the window (context lines are read-only)
                    ids = [i for i in it["evidence"]["utterance_ids"] if i in allowed]
                    if not ids:      # only context lines: the neighbouring window owns it
                        continue
                    it["evidence"]["utterance_ids"] = ids
                    items.append(it)
    return items


def extract_per_topic(utterances, meeting, title, attendees, effort, model=llm.EXTRACTOR,
                      skipped: list[dict] | None = None):
    ids = [u["id"] for u in utterances]
    if not ids:
        return [], []
    transcript = format_transcript(utterances, title, attendees)
    # the local model sees each line cut to 100 characters for segmentation: enough to place
    # topic boundaries, and a 90-minute meeting still fits its context
    seg_text = (format_transcript(utterances, title, attendees, max_chars=100)
                if llm.is_local(model) else transcript)
    seg, _ = llm.parse(Segmentation, model=model, system=prompts.SEGMENT_SYSTEM,
                       messages=[{"role": "user", "content": seg_text}],
                       purpose="segment", meeting=meeting, effort=effort)
    topics = _fix_spans(seg.topics, ids)
    for i, t in enumerate(topics, 1):
        t["id"] = f"t{i}"
    if llm.is_local(model):
        return topics, extract_local_windows(utterances, topics, meeting, title, attendees, model,
                                             skipped)

    cached_transcript = {"type": "text", "text": transcript, "cache_control": {"type": "ephemeral"}}
    local = llm.is_local(model)
    pos = {i: n for n, i in enumerate(ids)}

    def context_for(t) -> dict:
        """Claude reads the whole meeting for every topic (cached). A local 7B model gets the
        topic plus LOCAL_CONTEXT utterances either side: shorter, faster and more focused."""
        if not local:
            return cached_transcript
        a = max(0, pos[t["start_id"]] - LOCAL_CONTEXT)
        b = min(len(utterances), pos[t["end_id"]] + LOCAL_CONTEXT + 1)
        return {"type": "text", "text": format_transcript(utterances[a:b], title, attendees)}

    def one(t):
        ask = (f"Topic {t['id']}: {t['title']}\nExtract items from utterances "
               f"{t['start_id']} to {t['end_id']} only.")
        ext, _ = llm.parse(TopicExtraction, model=model, system=prompts.EXTRACT_SYSTEM,
                           messages=[{"role": "user", "content": [context_for(t),
                                                                  {"type": "text", "text": ask}]}],
                           purpose="extract", meeting=meeting, effort=effort, max_tokens=32000)
        return _flatten(ext, t["id"])

    # First topic alone writes the transcript cache; the rest read it in parallel. One GPU
    # serves the local model, so it gets one request at a time.
    results = [one(topics[0])]
    with ThreadPoolExecutor(max_workers=1 if local else 4) as pool:
        results += list(pool.map(one, topics[1:]))
    items = [it for r in results for it in r]
    return topics, items


def extract_single_pass(utterances, meeting, title, attendees, effort, model=llm.EXTRACTOR):
    transcript = format_transcript(utterances, title, attendees)
    mi, _ = llm.parse(MeetingInsights, model=model, system=prompts.SINGLE_PASS_SYSTEM,
                      messages=[{"role": "user", "content": transcript}],
                      purpose="single_pass", meeting=meeting, effort=effort, max_tokens=32000)
    topics, items = [], _flatten(mi)
    for i, t in enumerate(mi.topics, 1):
        tid = f"t{i}"
        topics.append({"id": tid, "title": t.title, "start_id": t.start_id, "end_id": t.end_id})
        items += [{"type": "key_point", "topic_id": tid, **kp.model_dump()} for kp in t.key_points]
    return topics, items, mi.summary


def summarize(topics, items, meeting, effort, model=llm.EXTRACTOR) -> str:
    kept = [it for it in items if it["verification"]["status"] != "rejected"]
    from pipeline.insights.verify import render
    by_topic = "\n\n".join(
        f"## {t['title']}\n" + "\n".join(f"- {render(it)}" for it in kept if it["topic_id"] == t["id"])
        for t in topics)
    s, _ = llm.parse(Summary, model=model, system=prompts.SUMMARY_SYSTEM,
                     messages=[{"role": "user", "content": by_topic}],
                     purpose="summarize", meeting=meeting, effort=effort)
    return s.summary


def prompt_version(model: str) -> str | int:
    """Part of the insights cache key: the local path has its own prompts and version."""
    if llm.is_local(model):
        return f"{prompts.PROMPT_VERSION}+local{prompts.LOCAL_PROMPT_VERSION}"
    return prompts.PROMPT_VERSION


def extract(utterances: list[dict], meeting: str, title: str = "Meeting",
            attendees: list[str] | None = None, mode: str = "per_topic",
            effort: str = "medium", extractor: str | None = None) -> dict:
    """Cited notes. `extractor` picks the model that segments, extracts and summarizes
    (default Opus); verification always uses the verifier model."""
    model = extractor or llm.EXTRACTOR
    skipped: list[dict] = []
    ledger_start = llm.spent(meeting=meeting)  # per meeting: safe with concurrent runs
    if mode == "per_topic":
        topics, items = extract_per_topic(utterances, meeting, title, attendees, effort, model,
                                          skipped)
        _number(items)
        verify(items, utterances, meeting)
        summary = summarize(topics, items, meeting, effort, model)
    elif mode == "single_pass":
        topics, items, summary = extract_single_pass(utterances, meeting, title, attendees, effort,
                                                     model)
        _number(items)
        verify(items, utterances, meeting)
    else:
        raise ValueError(mode)

    status = {s: sum(it["verification"]["status"] == s for it in items)
              for s in ("verified", "needs_review", "rejected")}
    return {
        "meta": {"mode": mode, "extractor": model, "verifier": llm.VERIFIER,
                 "effort": effort, "prompt_version": prompt_version(model),
                 "schema_version": SCHEMA_VERSION, "items": len(items), "status": status,
                 "cost_usd": round(llm.spent(meeting=meeting) - ledger_start, 4),
                 **({"skipped_windows": skipped} if skipped else {})},
        "summary": summary,
        "topics": topics,
        "items": items,
    }
