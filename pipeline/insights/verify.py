"""Two-layer verification of extracted items.

Layer 1 (deterministic): cited IDs exist; a metric's value appears in its cited text after
number normalization; an owner is a known speaker or a name said in the meeting; numbers the
recognizer was unsure about are flagged.
Layer 2 (semantic): a different model (VERIFIER) judges whether the cited lines support each item.

Status: rejected (hard check failed or unsupported; dropped from notes, kept in the log),
needs_review (soft flag or partial), verified.
"""

from __future__ import annotations

from pipeline import llm
from pipeline.insights.prompts import VERIFY_SYSTEM
from pipeline.insights.schema import Verification
from pipeline.numbers import numbers_in, value_in

GROUP_OWNERS = {"everyone", "all", "team", "the team", "group", "the group", "everybody"}
LOW_CONFIDENCE = 0.6


def render(item: dict) -> str:
    t = item["type"]
    if t == "action":
        return (f"Action item: {item['task']} | owner: {item.get('owner') or 'unassigned'} | "
                f"due: {item.get('due') or 'not stated'}")
    if t == "metric":
        unit = f" {item['unit']}" if item.get("unit") else ""
        ctx = f" ({item['context']})" if item.get("context") else ""
        return f"Metric: {item['name']} = {item['value']}{unit}{ctx}"
    if t == "qa":
        a = (f"answered by {item.get('answered_by') or 'someone'}: {item['answer']}"
             if item.get("answer") else "no answer")
        return (f"Question ({item['status']}) asked by {item.get('asked_by') or 'someone'}: "
                f"{item['question']} | {a}")
    label = {"decision": "Decision", "key_point": "Key point", "open_question": "Open question"}[t]
    return f"{label}: {item['text']}"


def deterministic_checks(item: dict, utts: dict[str, dict], speakers: set[str],
                         transcript_lower: str) -> list[dict]:
    ids = item["evidence"]["utterance_ids"]
    checks = []
    missing = [i for i in ids if i not in utts]
    checks.append({"check": "ids_exist", "ok": bool(ids) and not missing, "hard": True,
                   "detail": f"missing {missing}" if missing else ("no citations" if not ids else "")})
    cited = [utts[i] for i in ids if i in utts]

    if item["type"] == "metric":
        if numbers_in(item["value"]):
            ok = value_in(item["value"], [n for u in cited for n in numbers_in(u["text"])])
            checks.append({"check": "metric_value_in_text", "ok": ok, "hard": True,
                           "detail": "" if ok else f"value {item['value']!r} not in cited text"})
            unsure = [w["word"].strip() for u in cited for w in u.get("words", [])
                      if numbers_in(w["word"]) and w.get("prob", 1) < LOW_CONFIDENCE]
            checks.append({"check": "number_confidence", "ok": not unsure, "hard": False,
                           "detail": f"low-confidence words {unsure}" if unsure else ""})

    people = ([("owner", item.get("owner"))] if item["type"] == "action" else
              [("asked_by", item.get("asked_by")), ("answered_by", item.get("answered_by"))]
              if item["type"] == "qa" else [])
    for role, who in people:
        if not who:
            continue
        who = who.strip()
        ok = who in speakers or who.lower() in GROUP_OWNERS or who.lower() in transcript_lower
        checks.append({"check": f"{role}_known", "ok": ok, "hard": False,
                       "detail": "" if ok else f"{role.replace('_', ' ')} {who!r} is not a "
                                               "speaker or a name said"})
    return checks


LOCAL_BATCH = 15   # a 7B model judges a few items per call more reliably than ~100 at once


def semantic_verdicts(items: list[dict], utts: dict[str, dict], meeting: str) -> dict[str, dict]:
    if not items:
        return {}
    if llm.is_local(llm.VERIFIER) and len(items) > LOCAL_BATCH:
        out: dict[str, dict] = {}
        for i in range(0, len(items), LOCAL_BATCH):
            out.update(semantic_verdicts(items[i:i + LOCAL_BATCH], utts, meeting))
        return out
    blocks = []
    for it in items:
        lines = "\n".join(f"  [{i} | {utts[i]['speaker']}] {utts[i]['text']}"
                          for i in it["evidence"]["utterance_ids"] if i in utts)
        blocks.append(f"item_id: {it['id']}\n{render(it)}\ncited:\n{lines}")
    result, _ = llm.parse(
        Verification, model=llm.VERIFIER, system=VERIFY_SYSTEM,
        messages=[{"role": "user", "content": "\n\n".join(blocks)}],
        purpose="verify", meeting=meeting, effort="medium", max_tokens=32000,
    )
    return {v.item_id: {"support": v.support, "reason": v.reason} for v in result.verdicts}


def verify(items: list[dict], utterances: list[dict], meeting: str) -> list[dict]:
    """Annotate each item with item['verification'] = {checks, support, reason, status}."""
    utts = {u["id"]: u for u in utterances}
    speakers = {u["speaker"] for u in utterances}
    transcript_lower = " ".join(u["text"] for u in utterances).lower()

    for it in items:
        it["verification"] = {"checks": deterministic_checks(it, utts, speakers, transcript_lower)}
    candidates = [it for it in items
                  if all(c["ok"] or not c["hard"] for c in it["verification"]["checks"])]
    verdicts = semantic_verdicts(candidates, utts, meeting)
    # A local 7B judge wrongly calls correct notes "unsupported" (dev: 'microphone on the
    # side', in the human summary, rejected). It may flag a note for review, never delete it;
    # only the deterministic checks reject.
    weak_judge = llm.is_local(llm.VERIFIER)

    for it in items:
        v = it["verification"]
        hard_fail = any(not c["ok"] and c["hard"] for c in v["checks"])
        soft_flag = any(not c["ok"] and not c["hard"] for c in v["checks"])
        verdict = verdicts.get(it["id"])
        v["support"] = verdict["support"] if verdict else None
        v["reason"] = verdict["reason"] if verdict else None
        if hard_fail or (verdict and verdict["support"] == "unsupported" and not weak_judge):
            v["status"] = "rejected"
        elif verdict and verdict["support"] == "unsupported":   # weak judge: flag, don't delete
            v["status"] = "needs_review"
        elif verdict is None:
            v["status"] = "needs_review"  # verifier skipped it: never silently trust
        elif soft_flag or verdict["support"] == "partial":
            v["status"] = "needs_review"
        else:
            v["status"] = "verified"
    return items
