"""Score extracted insights against AMI's human summaries.

Metrics (per meeting, then pooled):
- recall: share of human decisions / actions / problems our notes cover (judge: covered,
  partial or missed). Reported strict (covered) and lenient (covered + partial).
- precision_lb: share of our decisions / action items that the judge matched to some human
  sentence. A LOWER BOUND: AMI summaries are incomplete, so correct items can go unmatched.
- citation_agreement: for matched (human sentence, our item) pairs where the human sentence has
  links, share where our cited utterances overlap in time with a human-linked dialogue act.
- wrong_number_rate: share of numeric metrics whose value does not appear in AMI's human
  transcript within the cited time window (+-2 s). No LLM involved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from evaluation.ami_summaries import human_summary
from pipeline import ami, llm
from pipeline.insights.verify import render
from pipeline.numbers import numbers_in, value_in

JUDGE_VERSION = 1
JUDGE_SYSTEM = """You compare meeting notes produced by a system against a human-written \
reference summary of the same meeting.

For each reference sentence, decide whether the system notes convey it:
- covered: one or more system items state the same thing (wording may differ; it may be split \
across items).
- partial: system items convey part of it but miss or contradict an important element \
(e.g. who, what was decided, the condition).
- missed: no system item conveys it.
List the item_ids you relied on (empty for missed) and give a one-sentence reason.
Judge meaning, not wording. Speaker labels in the system notes are anonymous \
(e.g. SPEAKER_02), while the reference uses roles (e.g. "the Marketing Expert"): do not count \
a role/label difference as an error."""

class Match(BaseModel):
    human_id: str
    status: Literal["covered", "partial", "missed"]
    item_ids: list[str]
    reason: str


class Matching(BaseModel):
    matches: list[Match]


def judge_version(model: str) -> int:
    from evaluation.local_judge import LOCAL_JUDGE_VERSION

    return LOCAL_JUDGE_VERSION if llm.is_local(model) else JUDGE_VERSION


def judge_filename(cache_id: str, model: str, version: int | None = None) -> str:
    """Cache file for one judge run. The Claude judge keeps its original name
    (judge.j1.<id>.json); a local judge's name carries the model, e.g.
    judge.j2-qwen2.5-7b.<id>.json, so its verdicts can never be mistaken for Sonnet's."""
    if not llm.is_local(model):
        return f"judge.j{JUDGE_VERSION if version is None else version}.{cache_id}.json"
    v = judge_version(model) if version is None else version
    return f"judge.j{v}-{llm.stage_suffix(model)}.{cache_id}.json"


def judge(meeting: str, insights: dict, human: list[dict], tag: str,
          model: str | None = None, version: int | None = None) -> list[dict]:
    """One {human_id, status, item_ids, reason} per reference sentence. model defaults to
    llm.VERIFIER (Sonnet with NOTES_LLM=claude, the local model with NOTES_LLM=local)."""
    model = model or llm.VERIFIER
    if llm.is_local(model):
        from evaluation.local_judge import judge_local

        return judge_local(meeting, insights, human, tag, model,
                           judge_version(model) if version is None else version)
    items = [it for it in insights["items"] if it["verification"]["status"] != "rejected"]
    notes = "\n".join(f"{it['id']}: {render(it)}" for it in items)
    refs = "\n".join(f"{h['id']} [{h['section']}]: {h['text']}" for h in human)
    result, _ = llm.parse(
        Matching, model=model, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content":
                   f"SYSTEM NOTES\n{notes}\n\nREFERENCE SENTENCES\n{refs}\n\n"
                   "Return one match per reference sentence."}],
        purpose=f"judge:{tag}", meeting=meeting, effort="medium", max_tokens=32000)
    return [m.model_dump() for m in result.matches]


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def score_meeting(meeting: str, insights: dict, utterances: list[dict],
                  matches: list[dict]) -> dict:
    human = {h["id"]: h for h in human_summary(meeting)}
    utts = {u["id"]: u for u in utterances}
    items = {it["id"]: it for it in insights["items"] if it["verification"]["status"] != "rejected"}
    by_human = {m["human_id"]: m for m in matches if m["human_id"] in human}

    # recall per section
    recall = {}
    for sec in ("decisions", "actions", "problems"):
        ids = [h for h, v in human.items() if v["section"] == sec]
        st = [by_human.get(h, {"status": "missed"})["status"] for h in ids]
        recall[sec] = {"n": len(ids), "covered": st.count("covered"), "partial": st.count("partial")}

    # precision lower bound: our decisions / actions matched to any human sentence
    matched_items = {i for m in matches if m["status"] != "missed" for i in m["item_ids"]}
    precision = {}
    for kind in ("decision", "action"):
        ours = [i for i, it in items.items() if it["type"] == kind]
        precision[kind] = {"n": len(ours), "matched": sum(i in matched_items for i in ours)}

    # citation agreement with human links
    pairs = hits = 0
    for hid, m in by_human.items():
        spans = human[hid]["linked_spans"]
        if m["status"] == "missed" or not spans:
            continue
        for iid in m["item_ids"]:
            if iid not in items:
                continue
            cited = [(utts[u]["start"], utts[u]["end"])
                     for u in items[iid]["evidence"]["utterance_ids"] if u in utts]
            pairs += 1
            hits += any(_overlaps(c, s) for c in cited for s in spans)

    # wrong numbers vs the human transcript
    ref = ami.reference_segments(meeting)
    numeric = wrong = 0
    wrong_examples = []
    for it in items.values():
        if it["type"] != "metric" or not numbers_in(it["value"]):
            continue
        cited = [utts[u] for u in it["evidence"]["utterance_ids"] if u in utts]
        if not cited:
            continue
        lo, hi = min(u["start"] for u in cited) - 2, max(u["end"] for u in cited) + 2
        ref_numbers = [n for s in ref if s["start_time"] < hi and s["end_time"] > lo
                       for n in numbers_in(s["words"])]
        numeric += 1
        if not value_in(it["value"], ref_numbers):
            wrong += 1
            wrong_examples.append({"id": it["id"], "metric": render(it),
                                   "status": it["verification"]["status"]})

    return {"recall": recall, "precision_lb": precision,
            "citation": {"pairs": pairs, "hits": hits},
            "numbers": {"numeric_metrics": numeric, "wrong": wrong, "examples": wrong_examples},
            "status": insights["meta"]["status"], "cost_usd": insights["meta"]["cost_usd"]}


def pool(per_meeting: dict[str, dict]) -> dict:
    def ratio(a, b):
        return round(a / b, 4) if b else None

    out = {"meetings": len(per_meeting)}
    for sec in ("decisions", "actions", "problems"):
        n = sum(r["recall"][sec]["n"] for r in per_meeting.values())
        c = sum(r["recall"][sec]["covered"] for r in per_meeting.values())
        p = sum(r["recall"][sec]["partial"] for r in per_meeting.values())
        out[f"recall_{sec}"] = {"n": n, "strict": ratio(c, n), "lenient": ratio(c + p, n)}
    for kind in ("decision", "action"):
        n = sum(r["precision_lb"][kind]["n"] for r in per_meeting.values())
        k = sum(r["precision_lb"][kind]["matched"] for r in per_meeting.values())
        out[f"precision_lb_{kind}"] = {"n": n, "value": ratio(k, n)}
    pairs = sum(r["citation"]["pairs"] for r in per_meeting.values())
    hits = sum(r["citation"]["hits"] for r in per_meeting.values())
    out["citation_agreement"] = {"pairs": pairs, "value": ratio(hits, pairs)}
    num = sum(r["numbers"]["numeric_metrics"] for r in per_meeting.values())
    wrong = sum(r["numbers"]["wrong"] for r in per_meeting.values())
    out["wrong_number_rate"] = {"numeric_metrics": num, "value": ratio(wrong, num)}
    st = {s: sum(r["status"][s] for r in per_meeting.values())
          for s in ("verified", "needs_review", "rejected")}
    out["item_status"] = st
    out["insights_cost_usd"] = round(sum(r["cost_usd"] for r in per_meeting.values()), 4)
    return out
