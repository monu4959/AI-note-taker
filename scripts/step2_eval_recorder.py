"""Step 2 gate: does a desktop recording come out as well as uploading the meeting file?

A real call can't be scored (no reference transcript), so the recorder's two tracks are
built from an AMI meeting's per-person headset channels, where the truth is known:

  mic     = one participant's headset ("you")
  others  = the other three headsets summed (what Zoom/Teams would play)

Levels are set to what a laptop records, not AMI's quiet close-talk headsets: your voice at
-26 dBFS (speech, Windows mic AGC), the call at -20 dBFS, a mic noise floor of -65 dBFS.

Conditions, each mixed by the recorder's own code (pipeline/recording.py) and run through the
same pipeline as an upload:
  upload      the Mix-Headset file, the normal upload path (baseline, cached from Step 1)
  headphones  the two tracks as above
  speakers    no headphones: the call also reaches the mic from the laptop speakers, 60 ms
              later, 10 dB below the loopback (about as loud as your voice), 150 ms room tail

Scored: WER (ORC, speaker-agnostic), cpWER (who said what), DER, and whether the cluster the
recorder names as you really is you. With --app, the speakers condition is also pushed
through the real app path (recorder.finalize -> worker job, incl. notes) to time
stop -> notes ready. That run costs ~$0.40 of LLM; everything else is free (GPU only).

Usage: .venv/Scripts/python scripts/step2_eval_recorder.py [--meeting ES2004b] [--you A] [--app]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import meeteval
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

from pipeline import ami
from pipeline import recording as rec
from pipeline.run import MEETINGS, PipelineConfig, run

OUT = ami.ROOT / "artifacts" / "eval" / "step2"
NORM = EnglishTextNormalizer()
RNG = np.random.default_rng(2)
SR = rec.SR


YOU_DB, CALL_DB, NOISE_DB = -26.0, -20.0, -65.0


def level_to(x: np.ndarray, target_db: float) -> np.ndarray:
    """Scale so the speech (active frames) sits at target_db."""
    g = rec.speech_gain(rec.frame_db(x)) * 10 ** ((target_db - rec.TARGET_DB) / 20)
    return (x * g).astype(np.float32)


def tracks_for(meeting: str, you: str, speakers: bool) -> tuple[np.ndarray, np.ndarray]:
    """(mic, loopback) at laptop levels; with `speakers`, the call echoes into the mic."""
    hs = {a: level_to(w, YOU_DB) for a, w in headsets(meeting).items()}
    others = level_to(np.sum([w for a, w in hs.items() if a != you], axis=0), CALL_DB)
    mic = hs[you] + RNG.normal(size=len(others)).astype(np.float32) * 10 ** (NOISE_DB / 20)
    if speakers:
        mic = mic + room_echo(others)
    return mic.astype(np.float32), others


def headsets(meeting: str) -> dict[str, np.ndarray]:
    """AMI agent letter -> that person's headset channel (16 kHz mono float32)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(ami.ANNOTATIONS / "corpusResources" / "meetings.xml").getroot()
    m = next(x for x in root.iter("meeting") if x.get("observation") == meeting)
    out = {}
    for s in m.iter("speaker"):
        wav, sr = sf.read(ami.AUDIO / f"{meeting}.Headset-{s.get('channel')}.wav", dtype="float32")
        assert sr == SR
        out[s.get("nxt_agent")] = wav if wav.ndim == 1 else wav.mean(axis=1)
    n = min(len(w) for w in out.values())
    return {k: v[:n] for k, v in out.items()}


def room_echo(x: np.ndarray, delay_ms=60, level_db=-10.0) -> np.ndarray:
    """Laptop speakers -> laptop mic: delayed, quieter, short room tail (FFT convolution)."""
    from scipy.signal import fftconvolve

    n_ir = int(0.15 * SR)
    ir = RNG.normal(size=n_ir) * np.exp(-np.arange(n_ir) / (0.03 * SR))
    ir /= np.sqrt(np.sum(ir**2))
    y = fftconvolve(x, ir)[:len(x)] * 10 ** (level_db / 20)
    d = int(delay_ms * SR / 1000)
    return np.concatenate([np.zeros(d), y[:-d]]).astype(np.float32)


def write_tracks(d: Path, mic: np.ndarray, others: np.ndarray, others_offset_s=0.1) -> list:
    d.mkdir(parents=True, exist_ok=True)
    off = int(others_offset_s * SR)   # the loopback opened 100 ms after the mic, as on a laptop
    for name, x in (("mic-0.pcm", mic), ("others-0.pcm", others[off:])):
        (d / name).write_bytes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return [rec.Track("mic", d / "mic-0.pcm", 0), rec.Track("others", d / "others-0.pcm", off)]


def tag(you: str) -> str:
    return "" if you == "A" else f"-{you}"   # speaker A was the first run; keep its cache


def build(meeting: str, you: str, condition: str) -> tuple[Path, dict]:
    """Recorder track files for one condition, mixed into source.wav like a real recording."""
    d = OUT / f"{meeting}-{condition}{tag(you)}"
    src, levels = d / "source.wav", d / "levels.npz"
    if (d / "mix.json").exists() and json.loads((d / "mix.json").read_text()).get(
            "version") == rec.MIX_VERSION:
        return d, json.loads((d / "mix.json").read_text())
    mic, others = tracks_for(meeting, you, condition == "speakers")
    tracks = write_tracks(d / "tracks", mic, others)
    info = rec.mix(tracks, src, levels)
    (d / "mix.json").write_text(json.dumps(info, indent=1))
    for t in tracks:
        t.path.unlink()
    return d, info


def utterances(mid: str) -> list[dict]:
    return json.loads((MEETINGS / mid / "assign" / "utterances.json").read_text(encoding="utf-8"))


def diarization(mid: str) -> dict:
    return json.loads((MEETINGS / mid / "diarize" / "diarization.json").read_text(encoding="utf-8"))


def score(meeting: str, mid: str) -> dict:
    def norm(segs):
        return [dict(s, words=NORM(s["words"])) for s in segs if NORM(s["words"]).strip()]

    ref = meeteval.io.SegLST(norm(ami.reference_segments(meeting)))
    utts = utterances(mid)
    single = meeteval.io.SegLST(norm([{"session_id": meeting, "speaker": "hyp", "start_time": u["start"],
                                       "end_time": u["end"], "words": u["text"]} for u in utts]))
    diar = meeteval.io.SegLST(norm([{"session_id": meeting, "speaker": u["speaker"], "start_time": u["start"],
                                     "end_time": u["end"], "words": u["text"]} for u in utts]))
    wer = meeteval.wer.combine_error_rates(*meeteval.wer.orcwer(ref, single).values())
    cpwer = meeteval.wer.combine_error_rates(*meeteval.wer.cpwer(ref, diar).values())
    hyp = Annotation(uri=meeting)
    for t in diarization(mid)["regular"]:
        hyp[Segment(t["start"], t["end"])] = t["speaker"]
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)(
        ami.reference_rttm(meeting), hyp, uem=ami.uem(meeting))
    return {"wer": round(wer.error_rate, 4), "cpwer": round(cpwer.error_rate, 4),
            "der": round(der, 4), "speakers": len(hyp.labels())}


def owner_check(meeting: str, you: str, mid: str, levels: Path) -> dict:
    """Is the cluster the recorder would call 'you' the cluster that really is you?"""
    target = ami.speaker_map(meeting)[you]
    ref = ami.reference_rttm(meeting)
    turns = diarization(mid)["exclusive"]
    overlap: dict[str, float] = {}
    for t in turns:
        seg = Segment(t["start"], t["end"])
        overlap[t["speaker"]] = overlap.get(t["speaker"], 0.0) + sum(
            (seg & s).duration for s, _, lab in ref.itertracks(yield_label=True) if lab == target)
    truth = max(overlap, key=overlap.get)
    lv = np.load(levels)
    shares = rec.owner_shares([(t["start"], t["end"], t["speaker"]) for t in turns],
                              lv["mic_heard_db"], lv["oth_heard_db"], int(lv["lag"]))
    picked, _ = rec.owner_label(shares)
    return {"truth_cluster": truth, "picked": picked, "correct": picked == truth,
            "share_you": shares.get(truth, {}).get("share"),
            "max_share_others": max((d["share"] for lab, d in shares.items() if lab != truth),
                                    default=None)}


def app_run(meeting: str, d: Path) -> dict:
    """The speakers condition through the real app: finalize -> worker job -> notes."""
    import shutil

    from services import db, recorder

    rec_dir = recorder.RECORDINGS / f"eval-{meeting}"
    shutil.rmtree(rec_dir, ignore_errors=True)
    you = ARGS.you
    mic, others = tracks_for(meeting, you, speakers=True)
    tracks = write_tracks(rec_dir, mic, others)
    meta = {"id": rec_dir.name, "title": f"Simulated call: {meeting} (AMI, speaker {you} is you)",
            "attendees": [], "owner_name": None, "call_app": None,
            "started_at": "", "state": "stopped",
            "tracks": [{"file": t.path.name, "kind": t.kind, "device": "simulated",
                        "offset": t.offset} for t in tracks]}
    (rec_dir / "meta.json").write_text(json.dumps(meta))
    mid = db.create_recording(meta["title"], [], None, recorder.SOURCE_NAME)
    t0 = time.monotonic()
    recorder.finalize(rec_dir, mid)
    print(f"  app: meeting {mid} queued after {time.monotonic() - t0:.0f} s of mixing", flush=True)
    while True:
        job = db.latest_job(mid)
        if job and job["status"] in ("done", "failed"):
            break
        time.sleep(5)
    elapsed = time.monotonic() - t0
    m = db.get_meeting(mid)
    with db.connect() as conn:
        spk = conn.execute("SELECT cluster_label, display_name, method FROM speakers"
                           " WHERE meeting_id = %s AND method = 'mic'", (mid,)).fetchall()
    return {"meeting_id": mid, "job": job["status"], "error": job["error"],
            "stop_to_notes_s": round(elapsed), "duration_s": m["duration_s"],
            "ratio": round(elapsed / m["duration_s"], 3) if m["duration_s"] else None,
            "named_by_mic": spk}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meeting, you = ARGS.meeting, ARGS.you
    results = {"meeting": meeting, "you": you, "you_global": ami.speaker_map(meeting)[you],
               "conditions": {}}
    base = f"ami-{meeting}"
    run(base, ami.audio_path(meeting), PipelineConfig(insights=False, name_cues=False))
    results["conditions"]["upload"] = score(meeting, base)
    print("upload", results["conditions"]["upload"], flush=True)
    for cond in ("headphones", "speakers"):
        d, info = build(meeting, you, cond)
        mid = f"ami-{meeting}-rec-{cond}{tag(you)}"
        t0 = time.perf_counter()
        run(mid, d / "source.wav", PipelineConfig(insights=False, name_cues=False))
        r = {**score(meeting, mid), "owner": owner_check(meeting, you, mid, d / "levels.npz"),
             "echo": info["echo"], "mic_suppressed_share": info["mic_suppressed_share"],
             "pipeline_s": round(time.perf_counter() - t0)}
        results["conditions"][cond] = r
        print(cond, r, flush=True)
    if ARGS.app:
        results["app"] = app_run(meeting, OUT / f"{meeting}-speakers{tag(you)}")
        print("app", results["app"], flush=True)
    out = OUT / f"{meeting}-{you}.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meeting", default="ES2004b")
    p.add_argument("--you", default="A", help="AMI agent letter of the person recording")
    p.add_argument("--app", action="store_true")
    ARGS = p.parse_args()
    main()
