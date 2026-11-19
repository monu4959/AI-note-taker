"""Speaker identification: map anonymous diarization clusters (SPEAKER_00, ...) to people.

Signals, strongest first (design doc, "Speaker identification"):
  1. voiceprints  - cosine similarity between a cluster's embedding and enrolled people's
  2. name cues    - an LLM reads the transcript for introductions / direct address (name_cues.py)
  3. the user     - confirms or renames; confirming can enroll the voice (opt-in)

A name is only proposed above a threshold calibrated for precision on AMI meeting series
(scripts/step3_calibrate.py): a wrong name on an action item is worse than "Speaker 2".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# The embedding model behind the diarizer's per-cluster centroids (WeSpeaker ResNet34 inside
# pyannote community-1). Stored with every voiceprint: changing models means re-enrolling.
EMBEDDING_MODEL = "pyannote/speaker-diarization-community-1"

# Calibrated in scripts/step3_calibrate.py (see reports/step3.md). Clusters with less speech
# than MIN_SPEECH_S have noisy centroids and are never named by voice.
THRESHOLD = 0.70
MIN_SPEECH_S = 10.0
TOP_K = 3   # a person's score = mean of their best TOP_K voiceprints (several mics / rooms)


@dataclass
class Cluster:
    label: str
    embedding: np.ndarray   # L2-normalised
    speech_s: float         # exclusive speaking time


@dataclass
class Match:
    label: str
    person: str
    score: float
    runner_up: float        # best score for any other person (for review / margin)


def clusters_from_diarization(diarize_dir: Path) -> list[Cluster]:
    """Per-cluster centroids from the diarize stage. pyannote orders speaker_embeddings like
    speaker_diarization.labels(), i.e. sorted labels; NaN rows (no clean speech) are skipped."""
    path = diarize_dir / "embeddings.npy"
    if not path.exists() or not (diarize_dir / "diarization.json").exists():
        return []
    turns = json.loads((diarize_dir / "diarization.json").read_text())
    emb = np.load(path)
    labels = sorted({t["speaker"] for t in turns["regular"]})
    speech: dict[str, float] = {}
    for t in turns["exclusive"]:
        speech[t["speaker"]] = speech.get(t["speaker"], 0.0) + t["end"] - t["start"]
    out = []
    for label, e in zip(labels, emb, strict=False):
        if not np.all(np.isfinite(e)) or not np.any(e):
            continue
        out.append(Cluster(label, normalise(e), round(speech.get(label, 0.0), 2)))
    return out


def normalise(e: np.ndarray) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    return e / np.linalg.norm(e)


def score_matrix(clusters: list[Cluster], gallery: dict[str, list[np.ndarray]],
                 top_k: int = TOP_K) -> tuple[list[str], np.ndarray]:
    """S[i, j] = similarity of cluster i to person j: mean cosine of the person's best top_k
    voiceprints. Returns (people, S)."""
    people = sorted(gallery)
    S = np.full((len(clusters), len(people)), -1.0)
    for j, p in enumerate(people):
        V = np.stack([normalise(v) for v in gallery[p]])
        for i, c in enumerate(clusters):
            sims = np.sort(V @ c.embedding)[::-1]
            S[i, j] = float(sims[:top_k].mean())
    return people, S


def identify(clusters: list[Cluster], gallery: dict[str, list[np.ndarray]],
             threshold: float = THRESHOLD, min_speech_s: float = MIN_SPEECH_S,
             candidates: set[str] | None = None) -> list[Match]:
    """One-to-one assignment (Hungarian), so two clusters never share a name; keep only
    matches at or above the threshold. `candidates` limits people (e.g. the attendee list)."""
    if candidates is not None:
        gallery = {p: v for p, v in gallery.items() if p in candidates}
    eligible = [c for c in clusters if c.speech_s >= min_speech_s]
    if not eligible or not gallery:
        return []
    people, S = score_matrix(eligible, gallery)
    rows, cols = linear_sum_assignment(-S)
    out = []
    for i, j in zip(rows, cols, strict=True):
        if S[i, j] < threshold:
            continue
        others = np.delete(S[i], j)
        out.append(Match(eligible[i].label, people[j], round(float(S[i, j]), 4),
                         round(float(others.max()) if others.size else -1.0, 4)))
    return out
