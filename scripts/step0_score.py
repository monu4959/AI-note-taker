"""Step 0.7: score whatever Step 0 outputs exist.

- DER / JER  (diarization): pyannote.metrics, no collar, overlap scored, AMI UEM. This is the
  setting of pyannote's published community-1 AMI number (17.0%), on all 16 test meetings.
- WER        (what was said): meeteval ORC-WER with the hypothesis as ONE stream, i.e.
  speaker-agnostic; reference utterances may be matched in any order that respects stream order.
- cpWER      (who said what): meeteval cpWER, our diarized utterances vs reference speakers.
Text on both sides goes through Whisper's EnglishTextNormalizer (fillers removed, numbers and
spelling normalized) so formatting differences are not counted as errors.

Usage: .venv/Scripts/python scripts/step0_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import meeteval
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

from pipeline import ami

STEP0 = ami.ROOT / "artifacts" / "step0"
NORM = EnglishTextNormalizer()


def _norm_segments(segs: list[dict]) -> list[dict]:
    out = []
    for s in segs:
        words = NORM(s["words"])
        if words.strip():
            out.append(dict(s, words=words))
    return out


def score_diarization() -> dict | None:
    dia_dir = STEP0 / "diarization"
    meetings = [m for m in ami.meetings("test") if (dia_dir / f"{m}.rttm").exists()]
    if not meetings:
        return None
    der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)
    per = {}
    for m in meetings:
        ref, hyp, uem = ami.reference_rttm(m), ami.read_rttm(dia_dir / f"{m}.rttm", m), ami.uem(m)
        d = der(ref, hyp, uem=uem, detailed=True)
        jer(ref, hyp, uem=uem)
        total = d["total"]
        per[m] = {
            "der": round(d["diarization error rate"], 4),
            "false_alarm": round(d["false alarm"] / total, 4),
            "missed": round(d["missed detection"] / total, 4),
            "confusion": round(d["confusion"] / total, 4),
            "ref_speakers": len(ref.labels()),
            "hyp_speakers": len(hyp.labels()),
        }
    return {"meetings": len(meetings), "der": round(abs(der), 4), "jer": round(abs(jer), 4),
            "per_meeting": per}


def _asr_hyp_single_stream(m: str) -> list[dict]:
    asr = json.loads((STEP0 / "asr" / f"{m}.json").read_text())
    return [{"session_id": m, "speaker": "hyp", "start_time": s["start"], "end_time": s["end"],
             "words": s["text"]} for s in asr["segments"]]


def _utterance_hyp(m: str) -> list[dict]:
    utts = json.loads((STEP0 / "utterances" / f"{m}.json").read_text())
    return [{"session_id": m, "speaker": u["speaker"], "start_time": u["start"],
             "end_time": u["end"], "words": u["text"]} for u in utts]


def _er_dict(er) -> dict:
    return {"wer": round(er.error_rate, 4), "errors": er.errors, "ref_words": er.length,
            "ins": er.insertions, "del": er.deletions, "sub": er.substitutions}


def score_words(kind: str) -> dict | None:
    if kind == "wer":
        meetings = [m for m in ami.STEP0_ASR_MEETINGS if (STEP0 / "asr" / f"{m}.json").exists()]
        hyp_fn, metric = _asr_hyp_single_stream, meeteval.wer.orcwer
    else:
        meetings = [m for m in ami.STEP0_ASR_MEETINGS
                    if (STEP0 / "utterances" / f"{m}.json").exists()]
        hyp_fn, metric = _utterance_hyp, meeteval.wer.cpwer
    if not meetings:
        return None
    ref = meeteval.io.SegLST(_norm_segments([s for m in meetings for s in ami.reference_segments(m)]))
    hyp = meeteval.io.SegLST(_norm_segments([s for m in meetings for s in hyp_fn(m)]))
    per = metric(ref, hyp)
    total = meeteval.wer.combine_error_rates(*per.values())
    return {"meetings": len(meetings), **_er_dict(total),
            "per_meeting": {m: _er_dict(er) for m, er in sorted(per.items())}}


def main() -> None:
    results = {
        "diarization": score_diarization(),
        "wer_orc_speaker_agnostic": score_words("wer"),
        "cpwer": score_words("cpwer"),
    }
    (STEP0 / "scores.json").write_text(json.dumps(results, indent=1))
    for k, v in results.items():
        if v is None:
            print(f"{k}: (no outputs yet)")
        else:
            head = {kk: vv for kk, vv in v.items() if kk != "per_meeting"}
            print(k, head)


if __name__ == "__main__":
    main()
