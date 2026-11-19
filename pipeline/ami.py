"""AMI Meeting Corpus helpers: paths, meeting lists, reference diarization and audio loading.

Reference diarization and scoring regions come from pyannote's fork of the BUT
AMI-diarization-setup ("only_words" protocol), the setup behind pyannote's published DER.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline

ROOT = Path(__file__).resolve().parents[1]
AMI = ROOT / "data" / "ami"
SETUP = AMI / "setup"
AUDIO = AMI / "audio"

# Step 0 transcription subset: the setup's predefined test.mini list (IS1009a, ES2004a,
# TS3003a), plus EN2002a (the only non-scenario series, i.e. natural meetings) and ES2004b
# (a full-length scenario meeting; the mini three are short kickoff meetings).
STEP0_ASR_MEETINGS = ["IS1009a", "ES2004a", "TS3003a", "EN2002a", "ES2004b"]


def meetings(split: str = "test") -> list[str]:
    path = SETUP / "lists" / f"{split}.meetings.txt"
    return [m.strip() for m in path.read_text().splitlines() if m.strip()]


def audio_path(meeting: str) -> Path:
    return AUDIO / f"{meeting}.Mix-Headset.wav"


def load_audio(meeting: str) -> tuple[np.ndarray, int]:
    """Mono float32 waveform and sample rate (AMI Mix-Headset is 16 kHz mono)."""
    wav, sr = sf.read(audio_path(meeting), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def reference_rttm(meeting: str, split: str = "test") -> Annotation:
    ann = Annotation(uri=meeting)
    path = SETUP / "only_words" / "rttms" / split / f"{meeting}.rttm"
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0] != "SPEAKER":
            continue
        start, dur, spk = float(f[3]), float(f[4]), f[7]
        ann[Segment(start, start + dur)] = spk
    return ann


def uem(meeting: str, split: str = "test") -> Timeline:
    tl = Timeline(uri=meeting)
    path = SETUP / "uems" / split / f"{meeting}.uem"
    for line in path.read_text().splitlines():
        f = line.split()
        if f:
            tl.add(Segment(float(f[2]), float(f[3])))
    return tl


ANNOTATIONS = AMI / "annotations"
_NITE_ID = "{http://nite.sourceforge.net/}id"


def speaker_map(meeting: str) -> dict[str, str]:
    """AMI agent letter (A-D) -> global speaker ID used in the reference RTTMs (e.g. MEO015)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(ANNOTATIONS / "corpusResources" / "meetings.xml").getroot()
    for m in root.iter("meeting"):
        if m.get("observation") == meeting:
            return {s.get("nxt_agent"): s.get("global_name") for s in m.iter("speaker")}
    raise KeyError(meeting)


def reference_segments(meeting: str, punctuation: bool = False) -> list[dict]:
    """Manual transcript as utterance segments, using AMI's own segmentation.

    Kept: lexical words (<w>). Dropped: truncated word fragments (trunc="true", e.g. "Tu"),
    vocal sounds, disfluency markers, gaps and (unless `punctuation`) punctuation tokens.
    Segment times are the first/last kept word's times. Scoring uses punctuation=False;
    punctuation=True gives readable text for LLM input.
    """
    import re
    import xml.etree.ElementTree as ET

    spk = speaker_map(meeting)
    out = []
    for agent, global_name in sorted(spk.items()):
        words_xml = ANNOTATIONS / "words" / f"{meeting}.{agent}.words.xml"
        segs_xml = ANNOTATIONS / "segments" / f"{meeting}.{agent}.segments.xml"
        if not words_xml.exists():
            continue
        elems = list(ET.parse(words_xml).getroot())
        index = {e.get(_NITE_ID): i for i, e in enumerate(elems)}
        for seg in ET.parse(segs_xml).getroot().iter("segment"):
            child = seg.find("{http://nite.sourceforge.net/}child")
            ids = re.findall(r"id\(([^)]+)\)", child.get("href"))
            lo, hi = index[ids[0]], index[ids[-1]]
            kept = [
                e for e in elems[lo : hi + 1]
                if e.tag == "w" and e.get("trunc") != "true" and e.text
                and e.get("starttime") is not None
                and (punctuation or e.get("punc") != "true")
            ]
            lexical = [e for e in kept if e.get("punc") != "true"]
            if not lexical:
                continue
            text = ""
            for e in kept:
                tok = e.text.strip()
                text += tok if (e.get("punc") == "true" or not text) else " " + tok
            out.append({
                "session_id": meeting,
                "speaker": global_name,
                "start_time": float(lexical[0].get("starttime")),
                "end_time": float(lexical[-1].get("endtime")),
                "words": text,
            })
    return sorted(out, key=lambda s: (s["start_time"], s["speaker"]))


def write_rttm(ann: Annotation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        ann.write_rttm(fh)


def read_rttm(path: Path, uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    for line in path.read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            ann[Segment(start, start + dur)] = f[7]
    return ann
