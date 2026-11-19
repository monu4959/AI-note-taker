"""Step 1.3: run insights + judge on AMI meetings and pool the metrics.

  --source asr  our pipeline transcript (GPU stages must exist; see step1_prepare_eval.py)
  --source ref  AMI's human transcript and speakers (oracle), isolating ASR/diarization error

Insights and judge outputs are cached, so re-running only spends on what changed.
Usage: .venv/Scripts/python scripts/step1_eval_insights.py --tag dev-v1 --source asr IS1008d ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ami_summaries import human_summary, reference_utterances
from evaluation.insights_eval import judge, judge_filename, judge_version, pool, score_meeting
from pipeline import ami, llm
from pipeline.insights import prompts
from pipeline.insights.extract import extract, prompt_version
from pipeline.run import MEETINGS, PipelineConfig, insights_stage, run
from pipeline.stages import MeetingStore

EVAL = ami.ROOT / "artifacts" / "eval"


def insights_for(meeting: str, source: str, trial: int,
                 extractor: str | None = None) -> tuple[dict, list[dict], str]:
    """Returns (insights, utterances, cache_id identifying exactly these insights).

    A copy of every scored insights file is kept under artifacts/eval/, because the pipeline's
    insights stage directory is replaced whenever the prompt or config changes.
    """
    title = f"AMI {meeting} (product design team meeting)"
    if source == "asr":
        run(f"ami-{meeting}", ami.audio_path(meeting),
            PipelineConfig(title=title, insights_trial=trial, insights_extractor=extractor,
                           name_cues=False))
        store = MeetingStore(MEETINGS, f"ami-{meeting}")
        stage = insights_stage(extractor)
        insights = store.read_json(stage, "insights.json")
        fingerprint = store.manifest(stage)["fingerprint"]
        keep = EVAL / "asr" / meeting / f"insights.{fingerprint}.json"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_text(json.dumps(insights, indent=1))
        return insights, store.read_json("assign", "utterances.json"), fingerprint
    utts = reference_utterances(meeting)
    suffix = f".t{trial}" if trial else ""
    if extractor:
        raise SystemExit("--extractor is only supported with --source asr")
    model = llm.EXTRACTOR
    # Opus keeps the original file names; any other model gets its own (never shared)
    tag = (f"p{prompts.PROMPT_VERSION}" if model == llm.CLAUDE_EXTRACTOR
           else f"p{prompt_version(model)}-{llm.stage_suffix(model)}")
    path = EVAL / "ref" / meeting / f"insights.{tag}{suffix}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(extract(utts, f"ref-{meeting}", title, extractor=model),
                                   indent=1))
    return json.loads(path.read_text()), utts, f"ref-{tag}{suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--source", choices=["asr", "ref"], default="asr")
    ap.add_argument("--trial", type=int, default=0, help=">0: fresh extraction, same prompt")
    ap.add_argument("--extractor", default=None,
                    help="model that writes the notes, e.g. claude-sonnet-5-5 (default: Opus)")
    ap.add_argument("meetings", nargs="+")
    a = ap.parse_args()

    llm.RUN_TAG = a.tag
    spend0 = llm.spent(run=a.tag)
    per = {}
    for m in a.meetings:
        insights, utts, cache_id = insights_for(m, a.source, a.trial, a.extractor)
        jpath = EVAL / a.source / m / judge_filename(cache_id, llm.VERIFIER)
        if not jpath.exists():
            jpath.parent.mkdir(parents=True, exist_ok=True)
            jpath.write_text(json.dumps(judge(m, insights, human_summary(m), a.tag), indent=1))
        per[m] = score_meeting(m, insights, utts, json.loads(jpath.read_text()))
        print(f"{m}: recall dec {per[m]['recall']['decisions']} act {per[m]['recall']['actions']} "
              f"| status {per[m]['status']}", flush=True)

    pooled = pool(per)
    record = {"tag": a.tag, "source": a.source, "trial": a.trial, "meetings": a.meetings,
              "prompt_version": prompts.PROMPT_VERSION, "judge_version": judge_version(llm.VERIFIER),
              "extractor": a.extractor or llm.EXTRACTOR, "verifier_judge": llm.VERIFIER,
              "created_at": datetime.now(UTC).isoformat(),
              "run_spend_usd": round(llm.spent(run=a.tag) - spend0, 4), "pooled": pooled,
              "per_meeting": per}
    (EVAL / "runs").mkdir(parents=True, exist_ok=True)
    (EVAL / "runs" / f"{a.tag}.json").write_text(json.dumps(record, indent=1))
    with (EVAL / "eval_runs.jsonl").open("a") as fh:
        fh.write(json.dumps({k: v for k, v in record.items() if k != "per_meeting"}) + "\n")
    print(json.dumps(pooled, indent=1))
    print(f"run spend ${record['run_spend_usd']:.3f} | {llm.BUDGET_SCOPE} total ${llm.spent():.3f}")


if __name__ == "__main__":
    main()
