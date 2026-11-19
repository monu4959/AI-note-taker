"""Speaker diarization behind a small interface."""

from __future__ import annotations

import gc
from typing import Protocol

import numpy as np
from pyannote.core import Annotation


class Diarizer(Protocol):
    name: str

    def params(self) -> dict: ...
    def diarize(self, wav: np.ndarray, sample_rate: int, num_speakers: int | None = None,
                min_speakers: int | None = None, max_speakers: int | None = None) -> dict:
        """Return {"regular": Annotation, "exclusive": Annotation, "embeddings": ndarray|None}."""
        ...
    def close(self) -> None: ...


class PyannoteDiarizer:
    """pyannote community-1. Step 0 reproduced its published AMI DER (17.05% vs 17.0%)."""

    name = "pyannote"

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1"):
        self.model_name = model
        self._pipe = None

    def params(self) -> dict:
        return {"impl": self.name, "model": self.model_name}

    def _load(self):
        if self._pipe is None:
            import torch
            from pyannote.audio import Pipeline

            import pipeline  # noqa: F401  (.env, CUDA DLLs)

            self._pipe = Pipeline.from_pretrained(self.model_name)
            self._pipe.to(torch.device("cuda"))
        return self._pipe

    def diarize(self, wav, sample_rate, num_speakers=None, min_speakers=None,
                max_speakers=None) -> dict:
        import torch

        hints = {k: v for k, v in {"num_speakers": num_speakers, "min_speakers": min_speakers,
                                   "max_speakers": max_speakers}.items() if v}
        out = self._load()({"waveform": torch.from_numpy(wav)[None], "sample_rate": sample_rate},
                           **hints)
        return {"regular": out.speaker_diarization,
                "exclusive": out.exclusive_speaker_diarization,
                "embeddings": out.speaker_embeddings}

    def close(self) -> None:
        import torch

        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()


def annotation_to_list(ann: Annotation) -> list[dict]:
    return [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": spk}
            for s, _, spk in ann.itertracks(yield_label=True)]


def list_to_annotation(turns: list[dict], uri: str = "meeting") -> Annotation:
    from pyannote.core import Segment

    ann = Annotation(uri=uri)
    for t in turns:
        ann[Segment(t["start"], t["end"])] = t["speaker"]
    return ann
