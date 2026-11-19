"""AMI human abstractive summaries (decisions / actions / problems) and their links to the
dialogue acts a human chose as evidence, as time spans."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache

from pipeline import ami

NS = "{http://nite.sourceforge.net/}"
SECTIONS = ("decisions", "actions", "problems")
_NA = {"NA", "N/A", "NONE"}


@lru_cache(maxsize=64)
def _word_times(meeting: str, agent: str) -> tuple[list, dict]:
    path = ami.ANNOTATIONS / "words" / f"{meeting}.{agent}.words.xml"
    elems = list(ET.parse(path).getroot())
    return elems, {e.get(NS + "id"): i for i, e in enumerate(elems)}


def _span(href: str) -> tuple[float, float] | None:
    """'ES2004a.B.words.xml#id(X.words4)..id(X.words9)' -> (start, end) from timed words."""
    m = re.match(r"(\w+)\.(\w)\.words\.xml#(.*)", href)
    meeting, agent, rng = m.group(1), m.group(2), m.group(3)
    elems, index = _word_times(meeting, agent)
    ids = re.findall(r"id\(([^)]+)\)", rng)
    lo, hi = index[ids[0]], index[ids[-1]]
    times = [(float(e.get("starttime")), float(e.get("endtime"))) for e in elems[lo:hi + 1]
             if e.get("starttime") is not None and e.get("endtime") is not None]
    return (times[0][0], times[-1][1]) if times else None


@lru_cache(maxsize=64)
def _dact_spans(meeting: str) -> dict[str, tuple[float, float]]:
    spans = {}
    for agent in "ABCDE":
        path = ami.ANNOTATIONS / "dialogueActs" / f"{meeting}.{agent}.dialog-act.xml"
        if not path.exists():
            continue
        for d in ET.parse(path).getroot().iter("dact"):
            child = d.find(NS + "child")
            if child is not None:
                s = _span(child.get("href"))
                if s:
                    spans[d.get(NS + "id")] = s
    return spans


def human_summary(meeting: str) -> list[dict]:
    """Non-NA sentences from the decisions/actions/problems sections, with linked spans."""
    root = ET.parse(ami.ANNOTATIONS / "abstractive" / f"{meeting}.abssumm.xml").getroot()
    links: dict[str, list[str]] = {}
    link_xml = ami.ANNOTATIONS / "extractive" / f"{meeting}.summlink.xml"
    for sl in ET.parse(link_xml).getroot().iter("summlink"):
        ext = abs_ = None
        for p in sl.iter(NS + "pointer"):
            target = re.search(r"#id\(([^)]+)\)", p.get("href")).group(1)
            if p.get("role") == "extractive":
                ext = target
            elif p.get("role") == "abstractive":
                abs_ = target
        if ext and abs_:
            links.setdefault(abs_, []).append(ext)

    dacts = _dact_spans(meeting)
    out = []
    for section in root:
        if section.tag not in SECTIONS:
            continue
        for sent in section.iter("sentence"):
            text = (sent.text or "").strip()
            if text.rstrip(".").upper() in _NA:
                continue
            sid = sent.get(NS + "id")
            out.append({"id": f"h{len(out) + 1}", "section": section.tag, "text": text,
                        "linked_spans": [dacts[d] for d in links.get(sid, []) if d in dacts]})
    return out


def reference_utterances(meeting: str) -> list[dict]:
    """AMI's human transcript in the pipeline's utterance format (for the oracle-transcript run)."""
    segs = ami.reference_segments(meeting, punctuation=True)
    return [{"id": f"u{i:04d}", "speaker": s["speaker"], "start": s["start_time"],
             "end": s["end_time"], "text": s["words"], "words": []}
            for i, s in enumerate(segs)]
