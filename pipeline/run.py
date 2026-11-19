"""Post-meeting pipeline: prepare -> transcribe -> diarize -> assign (-> insights, Step 1.2).

Usage: .venv/Scripts/python -m pipeline.run MEETING_ID AUDIO_PATH [--speakers N]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pipeline import audio
from pipeline.asr import FasterWhisperASR
from pipeline.assign import assign_words, to_utterances
from pipeline.diarization import PyannoteDiarizer, annotation_to_list, list_to_annotation
from pipeline.stages import MeetingStore, file_sha256, run_stage

ROOT = Path(__file__).resolve().parents[1]
MEETINGS = ROOT / "artifacts" / "meetings"


@dataclass
class PipelineConfig:
    asr: dict = field(default_factory=lambda: {"model": "large-v3", "compute_type": "int8_float16",
                                               "batch_size": 8, "beam_size": 5})
    diarizer: dict = field(default_factory=lambda: {
        "model": "pyannote/speaker-diarization-community-1"})
    num_speakers: int | None = None   # exact count, e.g. from the attendee list
    min_speakers: int | None = None
    max_speakers: int | None = None
    hotwords: str | None = None       # attendee names / glossary, passed to the recognizer
    normalize_loudness: bool = True
    title: str = "Meeting"
    attendees: list[str] | None = None
    insights: bool = True
    insights_mode: str = "per_topic"  # or "single_pass"
    insights_effort: str = "medium"
    insights_trial: int = 0           # >0 forces a fresh extraction (run-to-run variance)
    insights_extractor: str | None = None  # None = llm.EXTRACTOR (Opus); e.g. claude-sonnet-5-5
    name_cues: bool = True            # Step 3: names from introductions / direct address (LLM)


def run(meeting_id: str, source: Path, cfg: PipelineConfig | None = None,
        root: Path = MEETINGS, on_event: Callable[[dict], None] | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    store = MeetingStore(root, meeting_id)
    keys: dict[str, str] = {}

    # 1. prepare: any format -> 16 kHz mono wav (+ loudness normalization)
    def prepare(out: Path):
        if cfg.normalize_loudness:
            return audio.prepare(source, out / "audio.wav")
        wav = audio.decode(source)
        import soundfile as sf
        sf.write(out / "audio.wav", wav, audio.SAMPLE_RATE, subtype="PCM_16")
        return {"source": source.name, "duration_s": round(len(wav) / audio.SAMPLE_RATE, 3)}

    keys["prepare"] = run_stage(
        store, "prepare", 1,
        {"source_sha256": file_sha256(source), "normalize_loudness": cfg.normalize_loudness},
        {}, prepare, on_event)
    wav_path = store.dir("prepare") / "audio.wav"

    # 2. transcribe
    def transcribe(out: Path):
        asr = FasterWhisperASR(**cfg.asr, hotwords=cfg.hotwords)
        try:
            result = asr.transcribe(audio.load(wav_path))
        finally:
            asr.close()  # free GPU memory before diarization
        (out / "asr.json").write_text(json.dumps(result), encoding="utf-8")
        return {"words": len(result["words"]), "segments": len(result["segments"])}

    keys["transcribe"] = run_stage(
        store, "transcribe", 1, {**cfg.asr, "hotwords": cfg.hotwords},
        {"prepare": keys["prepare"]}, transcribe, on_event)

    # 3. diarize
    def diarize(out: Path):
        dia = PyannoteDiarizer(**cfg.diarizer)
        try:
            r = dia.diarize(audio.load(wav_path), audio.SAMPLE_RATE, cfg.num_speakers,
                            cfg.min_speakers, cfg.max_speakers)
        finally:
            dia.close()
        turns = {"regular": annotation_to_list(r["regular"]),
                 "exclusive": annotation_to_list(r["exclusive"])}
        (out / "diarization.json").write_text(json.dumps(turns))
        if r["embeddings"] is not None:
            np.save(out / "embeddings.npy", r["embeddings"])
        return {"speakers": sorted({t["speaker"] for t in turns["regular"]})}

    keys["diarize"] = run_stage(
        store, "diarize", 1,
        {**cfg.diarizer, "num_speakers": cfg.num_speakers, "min_speakers": cfg.min_speakers,
         "max_speakers": cfg.max_speakers},
        {"prepare": keys["prepare"]}, diarize, on_event)

    # 4. assign words to speakers -> utterances (stable IDs u0000, u0001, ...)
    def assign(out: Path):
        words = store.read_json("transcribe", "asr.json")["words"]
        turns = list_to_annotation(store.read_json("diarize", "diarization.json")["exclusive"],
                                   uri=meeting_id)
        assigned = assign_words(words, turns)
        utts = to_utterances(assigned)
        (out / "utterances.json").write_text(json.dumps(utts), encoding="utf-8")
        return {"utterances": len(utts), "far_words": sum(w["far"] for w in assigned)}

    keys["assign"] = run_stage(
        store, "assign", 1, {}, {"transcribe": keys["transcribe"], "diarize": keys["diarize"]},
        assign, on_event)

    # 5. name cues: speaker names the transcript itself shows (LLM, then deterministic checks).
    # Voiceprint matching needs the enrolled-people gallery in Postgres, so it runs in the
    # service layer (services/identify.py), which combines both signals.
    if cfg.name_cues:
        from pipeline import llm, name_cues

        def names(out: Path):
            utts = store.read_json("assign", "utterances.json")
            cues, rec = name_cues.propose(utts, cfg.title, cfg.attendees, meeting=meeting_id)
            result = {"cues": [c.model_dump() for c in cues], **name_cues.resolve(cues, utts),
                      "cost_usd": rec["cost_usd"]}
            (out / "names.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            return {"accepted": len(result["accepted"]), "proposed": len(cues)}

        keys["names"] = run_stage(
            store, names_stage(), 1,
            {"model": llm.VERIFIER, "prompt_version": name_cues.PROMPT_VERSION,
             "title": cfg.title, "attendees": cfg.attendees},
            {"assign": keys["assign"]}, names, on_event)

    # 6. insights: cited extraction + verification (LLM; spend is logged and capped)
    if cfg.insights:
        from pipeline import llm
        from pipeline.insights.extract import extract, prompt_version
        from pipeline.insights.schema import SCHEMA_VERSION

        extractor = cfg.insights_extractor or llm.EXTRACTOR

        def insights(out: Path):
            utts = store.read_json("assign", "utterances.json")
            try:
                result = extract(utts, meeting_id, cfg.title, cfg.attendees, cfg.insights_mode,
                                 cfg.insights_effort, extractor)
            finally:
                if llm.is_local(extractor):   # give the GPU back for the next transcription
                    from pipeline import llm_local

                    llm_local.unload(extractor)
            (out / "insights.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
            return result["meta"]

        # Each extractor model gets its own stage directory, so notes from two models can
        # coexist for one meeting (Opus keeps the original "insights" directory and cache).
        keys["insights"] = run_stage(
            store, insights_stage(extractor), 1,
            {"mode": cfg.insights_mode, "effort": cfg.insights_effort,
             "extractor": extractor, "verifier": llm.VERIFIER,
             "prompt_version": prompt_version(extractor), "schema_version": SCHEMA_VERSION,
             "title": cfg.title, "attendees": cfg.attendees,
             **({"trial": cfg.insights_trial} if cfg.insights_trial else {})},
            {"assign": keys["assign"]}, insights, on_event)

    return {"meeting_id": meeting_id, "keys": keys, "config": asdict(cfg)}


def insights_stage(extractor: str | None) -> str:
    """Stage directory for notes written by `extractor` (default: the active provider's).
    Opus keeps the original "insights" directory, so cached Claude notes are never replaced
    by a local run (a stage directory is rewritten whenever its settings change)."""
    from pipeline import llm

    extractor = extractor or llm.EXTRACTOR
    if extractor == llm.CLAUDE_EXTRACTOR:
        return "insights"
    return "insights-" + llm.stage_suffix(extractor)


def names_stage() -> str:
    """Stage directory for transcript name cues from the active verifier model."""
    from pipeline import llm

    return "names" if llm.VERIFIER == llm.CLAUDE_VERIFIER else "names-" + llm.stage_suffix(llm.VERIFIER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting_id")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--speakers", type=int, default=None)
    a = ap.parse_args()
    run(a.meeting_id, a.audio, PipelineConfig(num_speakers=a.speakers), on_event=print)


if __name__ == "__main__":
    main()
