"""Step 0.6a: diarize the AMI test split with pyannote community-1.

Saves per meeting: regular RTTM (scored for DER), exclusive RTTM (used for word assignment),
speaker embeddings (for Step 3), and a timing/GPU-memory log.

Usage: .venv/Scripts/python scripts/step0_diarize.py [MEETING ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from pyannote.audio import Pipeline

import pipeline  # noqa: F401  (registers CUDA DLLs on Windows)
from pipeline import ami

MODEL = "pyannote/speaker-diarization-community-1"
OUT = ami.ROOT / "artifacts" / "step0" / "diarization"


def main(selected: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diarizer = Pipeline.from_pretrained(MODEL)
    diarizer.to(torch.device("cuda"))

    log_path = OUT / "timings.jsonl"
    for meeting in selected:
        if (OUT / f"{meeting}.rttm").exists():
            print(f"skip {meeting} (done)")
            continue
        wav, sr = ami.load_audio(meeting)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = diarizer({"waveform": torch.from_numpy(wav)[None], "sample_rate": sr})
        elapsed = time.perf_counter() - t0

        ami.write_rttm(out.speaker_diarization, OUT / f"{meeting}.rttm")
        ami.write_rttm(out.exclusive_speaker_diarization, OUT / f"{meeting}.exclusive.rttm")
        if out.speaker_embeddings is not None:
            np.save(OUT / f"{meeting}.embeddings.npy", out.speaker_embeddings)

        rec = {
            "meeting": meeting,
            "audio_s": round(len(wav) / sr, 1),
            "elapsed_s": round(elapsed, 1),
            "realtime_factor": round(elapsed / (len(wav) / sr), 4),
            "num_speakers": len(out.speaker_diarization.labels()),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec)


if __name__ == "__main__":
    main(sys.argv[1:] or ami.meetings("test"))
