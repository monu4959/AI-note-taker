"""Join ASR words to diarization turns and group them into utterances.

Rules (from the design doc, "Word assignment edge cases"):
- A word goes to the turn it overlaps most.
- A word overlapping no turn goes to the nearest turn; if that turn is more than
  GAP_TOLERANCE_S away, the word is flagged `far`.
- Consecutive words of one speaker form an utterance, split at pauses > MAX_PAUSE_S.
Utterances are the unit insights cite, so their order and IDs must be deterministic.
"""

from __future__ import annotations

import numpy as np
from pyannote.core import Annotation

GAP_TOLERANCE_S = 0.5
MAX_PAUSE_S = 1.0


def assign_words(words: list[dict], turns: Annotation) -> list[dict]:
    tracks = [(seg.start, seg.end, spk) for seg, _, spk in turns.itertracks(yield_label=True)]
    if not tracks:
        return [dict(w, speaker="UNKNOWN", far=True) for w in words]
    starts = np.array([t[0] for t in tracks])
    ends = np.array([t[1] for t in tracks])
    labels = [t[2] for t in tracks]

    out = []
    for w in words:
        overlap = np.minimum(ends, w["end"]) - np.maximum(starts, w["start"])
        best = int(np.argmax(overlap))
        if overlap[best] > 0:
            out.append(dict(w, speaker=labels[best], far=False))
            continue
        # no overlap: distance from the word to each turn (0 if touching)
        dist = np.maximum(starts - w["end"], w["start"] - ends).clip(min=0)
        best = int(np.argmin(dist))
        out.append(dict(w, speaker=labels[best], far=bool(dist[best] > GAP_TOLERANCE_S)))
    return out


def to_utterances(assigned: list[dict]) -> list[dict]:
    utts: list[dict] = []
    for w in sorted(assigned, key=lambda w: w["start"]):
        cur = utts[-1] if utts else None
        if cur and cur["speaker"] == w["speaker"] and w["start"] - cur["end"] <= MAX_PAUSE_S:
            cur["words"].append(w)
            cur["end"] = max(cur["end"], w["end"])
        else:
            utts.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "words": [w]})
    for i, u in enumerate(utts):
        u["id"] = f"u{i:04d}"
        u["text"] = "".join(x["word"] for x in u["words"]).strip()
    return utts
