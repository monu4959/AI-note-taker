"""Step 0.6c: join ASR words to diarization speakers and build utterances.

Uses the *exclusive* diarization (one speaker at a time), which pyannote provides
specifically for aligning with transcripts.

Usage: .venv/Scripts/python scripts/step0_assign.py [MEETING ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import ami
from pipeline.assign import assign_words, to_utterances

STEP0 = ami.ROOT / "artifacts" / "step0"
OUT = STEP0 / "utterances"


def main(selected: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for m in selected:
        words = json.loads((STEP0 / "asr" / f"{m}.json").read_text())["words"]
        turns = ami.read_rttm(STEP0 / "diarization" / f"{m}.exclusive.rttm", m)
        assigned = assign_words(words, turns)
        utts = to_utterances(assigned)
        (OUT / f"{m}.json").write_text(json.dumps(utts, indent=1))
        far = sum(w["far"] for w in assigned)
        print(f"{m}: {len(words)} words -> {len(utts)} utterances, "
              f"{len({u['speaker'] for u in utts})} speakers, {far} far words "
              f"({far / max(len(words), 1):.1%})")


if __name__ == "__main__":
    main(sys.argv[1:] or ami.STEP0_ASR_MEETINGS)
