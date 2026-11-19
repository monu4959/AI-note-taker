"""Step 0.6b: transcribe AMI meetings with faster-whisper large-v3 (word timestamps).

Batched inference (VAD-chunked) is faster than sequential decoding and avoids the
repetition loops Whisper is prone to on hour-long audio. int8_float16 keeps large-v3
around 2 GB of VRAM on the 8 GB laptop GPU.

Usage: .venv/Scripts/python scripts/step0_transcribe.py [MEETING ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faster_whisper import BatchedInferencePipeline, WhisperModel

import pipeline  # noqa: F401  (registers CUDA DLLs on Windows)
from pipeline import ami

MODEL = "large-v3"
COMPUTE_TYPE = "int8_float16"
BATCH_SIZE = 8
OUT = ami.ROOT / "artifacts" / "step0" / "asr"


def main(selected: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(MODEL, device="cuda", compute_type=COMPUTE_TYPE)
    batched = BatchedInferencePipeline(model=model)

    for meeting in selected:
        path = OUT / f"{meeting}.json"
        if path.exists():
            print(f"skip {meeting} (done)")
            continue
        wav, sr = ami.load_audio(meeting)
        assert sr == 16000, sr
        t0 = time.perf_counter()
        segments, info = batched.transcribe(
            wav, language="en", word_timestamps=True, batch_size=BATCH_SIZE, beam_size=5
        )
        words, segs = [], []
        for s in segments:  # generator: decoding happens while iterating
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            for w in s.words or []:
                words.append(
                    {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                )
        elapsed = time.perf_counter() - t0
        result = {
            "meeting": meeting,
            "model": MODEL,
            "compute_type": COMPUTE_TYPE,
            "batch_size": BATCH_SIZE,
            "audio_s": round(len(wav) / sr, 1),
            "elapsed_s": round(elapsed, 1),
            "realtime_factor": round(elapsed / (len(wav) / sr), 4),
            "segments": segs,
            "words": words,
        }
        path.write_text(json.dumps(result, indent=1))
        print(meeting, f"{len(words)} words", f"{elapsed:.0f}s", f"rtf={result['realtime_factor']}")


if __name__ == "__main__":
    main(sys.argv[1:] or ami.STEP0_ASR_MEETINGS)
