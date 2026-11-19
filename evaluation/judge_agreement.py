"""Agreement between two judges' verdicts on the same reference sentences (no LLM here).

A verdict is one of covered / partial / missed per human summary sentence. Views:
- 3-way:   the verdicts as they are.
- lenient: covered + partial vs missed (what lenient recall counts).
- strict:  covered vs partial + missed (what strict recall counts).
"""

from __future__ import annotations

from collections import Counter

STATUSES = ("covered", "partial", "missed")
SECTIONS = ("decisions", "actions", "problems")


def collapse(status: str, view: str) -> str:
    if view == "3way":
        return status
    if view == "lenient":
        return "missed" if status == "missed" else "found"
    if view == "strict":
        return "covered" if status == "covered" else "not"
    raise ValueError(view)


def statuses(matches: list[dict], human_ids: list[str]) -> list[str]:
    """One status per human id, in order; a sentence the judge skipped counts as missed
    (as score_meeting does)."""
    by_id = {m["human_id"]: m["status"] for m in matches}
    return [by_id.get(h, "missed") for h in human_ids]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Unweighted Cohen's kappa. None when undefined (no items, or both judges constant on
    the same single label, where chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError("label lists differ in length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n**2
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def confusion(a: list[str], b: list[str], labels=STATUSES) -> dict[str, dict[str, int]]:
    """rows = judge a, columns = judge b."""
    out = {x: dict.fromkeys(labels, 0) for x in labels}
    for x, y in zip(a, b, strict=True):
        out[x][y] += 1
    return out


def agreement(a: list[str], b: list[str]) -> dict:
    """Raw agreement and kappa for the 3-way, lenient and strict views, plus the confusion."""
    out: dict = {"n": len(a)}
    for view in ("3way", "lenient", "strict"):
        x = [collapse(s, view) for s in a]
        y = [collapse(s, view) for s in b]
        k = cohen_kappa(x, y)
        out[view] = {"agreement": round(sum(p == q for p, q in zip(x, y, strict=True)) / len(x), 4)
                     if x else None,
                     "kappa": None if k is None else round(k, 4)}
    out["confusion"] = confusion(a, b)
    return out


def recall(statuses_by_section: dict[str, list[str]]) -> dict:
    """{section: [status, ...]} -> {section: {n, strict, lenient}}."""
    out = {}
    for sec in SECTIONS:
        st = statuses_by_section.get(sec, [])
        n = len(st)
        c, p = st.count("covered"), st.count("partial")
        out[sec] = {"n": n, "strict": round(c / n, 4) if n else None,
                    "lenient": round((c + p) / n, 4) if n else None}
    return out
