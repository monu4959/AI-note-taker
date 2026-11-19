"""Stage 1 helper: decode any upload to 16 kHz mono float32 and normalize loudness.

PyAV (installed with faster-whisper) bundles ffmpeg, so mp3/m4a/mp4/webm/wav all decode
without a system ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TARGET_RMS_DBFS = -20.0
PEAK_LIMIT = 0.99


def decode(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode the first audio stream to mono float32 at `sample_rate`."""
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def normalize_loudness(wav: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale to a target RMS over non-silent samples, without clipping. Returns (wav, gain_db)."""
    voiced = wav[np.abs(wav) > 1e-4]
    if voiced.size == 0:
        return wav, 0.0
    rms = float(np.sqrt(np.mean(voiced**2)))
    gain = 10 ** (TARGET_RMS_DBFS / 20) / rms
    peak = float(np.max(np.abs(wav))) * gain
    if peak > PEAK_LIMIT:
        gain *= PEAK_LIMIT / peak
    return (wav * gain).astype(np.float32), float(20 * np.log10(gain))


def prepare(src: Path, dst: Path) -> dict:
    wav = decode(src)
    wav, gain_db = normalize_loudness(wav)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, wav, SAMPLE_RATE, subtype="PCM_16")
    return {"source": src.name, "duration_s": round(len(wav) / SAMPLE_RATE, 3),
            "sample_rate": SAMPLE_RATE, "gain_db": round(gain_db, 2)}


def load(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SAMPLE_RATE, sr
    return wav.mean(axis=1)
