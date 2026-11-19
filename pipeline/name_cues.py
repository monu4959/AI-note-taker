"""Speaker names from the transcript: self-introductions and direct address.

An LLM proposes (speaker, name, kind, evidence); deterministic checks then keep only what
the transcript shows:
  - the spoken name appears as a word in a cited utterance
  - self_introduction: that utterance is by the speaker being named
  - addressed: the utterance is by someone else, and the named speaker talks within
    REPLY_WINDOW utterances of it (they answer, or were just answered)
Conflicts abstain: a name proposed for two speakers, or two names for one speaker with equal
support, is dropped. Cue names never override a voiceprint match (see services/identify.py).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from pipeline import llm
from pipeline.insights.extract import format_transcript

PROMPT_VERSION = 1
REPLY_WINDOW = 2

SYSTEM = """You identify which named person each anonymous speaker label is in a meeting \
transcript. Speaker labels (SPEAKER_00, ...) come from automatic diarization and are \
occasionally wrong at turn boundaries.

Propose a name for a speaker label ONLY from direct evidence in the transcript:
- self_introduction: the speaker says their own name ("I'm Priya", "this is Raj speaking", \
"Priya here").
- addressed: another speaker addresses them by name and they respond, or a speaker answers \
right after being called on ("Raj, can you take that?" followed by Raj's label speaking; \
"Thanks, Priya" right after Priya's label spoke).

Do not guess from roles, topics, voice, or third-person mentions ("as Maya said"). A name \
mentioned while talking ABOUT someone is not evidence. If evidence is ambiguous, propose \
nothing for that label. It is fine to return few or no cues.

For each cue give: speaker label, the name exactly as written in the transcript \
(spoken_as), the display name (use the attendee list's spelling when it is clearly the same \
person, otherwise the spoken form, capitalised), the kind, and 1-3 utterance IDs whose text \
contains the name. Return every distinct piece of evidence as its own cue."""


class NameCue(BaseModel):
    speaker: str = Field(description="Speaker label, e.g. SPEAKER_03")
    name: str = Field(description="Display name")
    spoken_as: str = Field(description="The name exactly as it appears in the cited text")
    kind: Literal["self_introduction", "addressed"]
    utterance_ids: list[str]


class NameCues(BaseModel):
    cues: list[NameCue]


LOCAL_CHUNK = 250   # utterances per name-cue call for a local model


def propose(utterances: list[dict], title: str, attendees: list[str] | None,
            meeting: str | None = None) -> tuple[list[NameCue], dict]:
    if llm.is_local(llm.VERIFIER) and len(utterances) > LOCAL_CHUNK:
        # a 7B model's context can't hold a long meeting: propose per chunk, then merge
        # (resolve() already handles several cues for one speaker or name)
        cues, recs = [], []
        for i in range(0, len(utterances), LOCAL_CHUNK):
            c, r = propose(utterances[i:i + LOCAL_CHUNK], title, attendees, meeting)
            cues += c
            recs.append(r)
        return cues, {"cost_usd": sum(r["cost_usd"] for r in recs), "chunks": len(recs)}
    transcript = format_transcript(utterances, title, attendees)
    out, rec = llm.parse(
        NameCues, model=llm.VERIFIER, system=SYSTEM, effort="medium", max_tokens=8000,
        messages=[{"role": "user", "content": transcript}], purpose="name_cues",
        meeting=meeting)
    return out.cues, rec


def _has_word(text: str, word: str) -> bool:
    return bool(word) and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE) is not None


def check(cue: NameCue, utterances: list[dict]) -> str | None:
    """Return None if the cue is supported, else the reason it is rejected."""
    pos = {u["id"]: i for i, u in enumerate(utterances)}
    labels = {u["speaker"] for u in utterances}
    if cue.speaker not in labels:
        return "unknown speaker"
    cited = [pos[i] for i in cue.utterance_ids if i in pos]
    if not cited:
        return "no valid utterance ids"
    spoken = cue.spoken_as.strip()
    hits = [i for i in cited if _has_word(utterances[i]["text"], spoken)]
    if not hits:
        return "name not in cited text"
    for i in hits:
        by = utterances[i]["speaker"]
        if cue.kind == "self_introduction" and by == cue.speaker:
            return None
        if cue.kind == "addressed" and by != cue.speaker:
            lo, hi = max(0, i - REPLY_WINDOW), min(len(utterances), i + REPLY_WINDOW + 1)
            if any(utterances[j]["speaker"] == cue.speaker for j in range(lo, hi) if j != i):
                return None
    return ("not said by this speaker" if cue.kind == "self_introduction"
            else "named speaker does not talk next to it")


def _weight(kind: str) -> float:
    return 2.0 if kind == "self_introduction" else 1.0


def resolve(cues: list[NameCue], utterances: list[dict]) -> dict:
    """Checked, conflict-free names: {"accepted": {label: {...}}, "rejected": [...]}."""
    rejected, support = [], {}
    for c in cues:
        reason = check(c, utterances)
        if reason:
            rejected.append({**c.model_dump(), "reason": reason})
            continue
        key = (c.speaker, c.name.strip())
        s = support.setdefault(key, {"score": 0.0, "kinds": set(), "evidence": []})
        s["score"] += _weight(c.kind)
        s["kinds"].add(c.kind)
        s["evidence"] += [i for i in c.utterance_ids if i not in s["evidence"]]

    # one name per speaker: best-supported; a tie abstains
    by_speaker: dict[str, list] = {}
    for (spk, name), s in support.items():
        by_speaker.setdefault(spk, []).append((s["score"], name, s))
    chosen = {}
    for spk, opts in by_speaker.items():
        opts.sort(key=lambda o: -o[0])
        if len(opts) > 1 and opts[0][0] == opts[1][0]:
            rejected += [{"speaker": spk, "name": o[1], "reason": "tied with another name"}
                         for o in opts]
            continue
        chosen[spk] = opts[0]

    # one speaker per name: a name claimed by two speakers abstains for both
    names: dict[str, list[str]] = {}
    for spk, (_, name, _) in chosen.items():
        names.setdefault(name.lower(), []).append(spk)
    accepted = {}
    for spk, (score, name, s) in chosen.items():
        if len(names[name.lower()]) > 1:
            rejected.append({"speaker": spk, "name": name, "reason": "name claimed by two speakers"})
            continue
        accepted[spk] = {"name": name, "score": score, "kinds": sorted(s["kinds"]),
                         "evidence": s["evidence"]}
    return {"accepted": accepted, "rejected": rejected}
