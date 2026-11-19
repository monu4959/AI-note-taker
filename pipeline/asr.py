"""Speech recognition behind a small interface, so models can be swapped by config."""

from __future__ import annotations

import gc
from typing import Protocol

import numpy as np


class ASR(Protocol):
    name: str

    def params(self) -> dict: ...
    def transcribe(self, wav: np.ndarray) -> dict:
        """Return {"segments": [{start, end, text}], "words": [{start, end, word, prob}]}."""
        ...
    def close(self) -> None: ...


class FasterWhisperASR:
    """faster-whisper, batched (VAD-chunked) decoding. Step 0: batched beat sequential on both
    WER and speed on AMI ES2004a."""

    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", compute_type: str = "int8_float16",
                 batch_size: int = 8, beam_size: int = 5, language: str = "en",
                 hotwords: str | None = None):
        self.model_name, self.compute_type = model, compute_type
        self.batch_size, self.beam_size, self.language = batch_size, beam_size, language
        self.hotwords = hotwords  # attendee names / glossary as vocabulary hints
        self._model = self._batched = None

    def params(self) -> dict:
        return {"impl": self.name, "model": self.model_name, "compute_type": self.compute_type,
                "batch_size": self.batch_size, "beam_size": self.beam_size,
                "language": self.language, "hotwords": self.hotwords}

    def _load(self):
        if self._batched is None:
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            import pipeline  # noqa: F401  (CUDA DLLs on Windows)

            self._model = WhisperModel(self.model_name, device="cuda",
                                       compute_type=self.compute_type)
            self._batched = BatchedInferencePipeline(model=self._model)
        return self._batched

    def transcribe(self, wav: np.ndarray) -> dict:
        segments, _ = self._load().transcribe(
            wav, language=self.language, word_timestamps=True, batch_size=self.batch_size,
            beam_size=self.beam_size, hotwords=self.hotwords,
        )
        segs, words = [], []
        for s in segments:
            segs.append({"start": s.start, "end": s.end, "text": s.text})
            words += [{"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                      for w in s.words or []]
        return {"segments": segs, "words": words}

    def close(self) -> None:
        self._batched = self._model = None
        gc.collect()
