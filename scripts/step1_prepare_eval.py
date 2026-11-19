"""Step 1.3 prep: run the GPU stages (prepare, transcribe, diarize, assign) on the insight-eval
meetings. No LLM calls. Re-runs are cached.

Usage: .venv/Scripts/python scripts/step1_prepare_eval.py [MEETING ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import ami
from pipeline.run import PipelineConfig, run

DEV = ["IS1008d", "ES2011b", "TS3004c"]
TEST = [f"{s}{x}" for s in ("IS1009", "ES2004", "TS3003") for x in "abcd"]


def main(meetings: list[str]) -> None:
    for m in meetings:
        t0 = time.perf_counter()
        run(f"ami-{m}", ami.audio_path(m), PipelineConfig(insights=False, name_cues=False))
        print(f"{m} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DEV + TEST)
