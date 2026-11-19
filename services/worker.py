"""GPU worker: claims queued jobs from Postgres and runs the post-meeting pipeline.

Runs natively (not in Docker) because it needs the Windows GPU. One worker is enough for a
laptop; several can run safely thanks to FOR UPDATE SKIP LOCKED.

Usage: .venv/Scripts/python -m services.worker
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from pipeline.run import MEETINGS, PipelineConfig, insights_stage, run
from pipeline.stages import MeetingStore
from services import db, identify

log = logging.getLogger("worker")

# Rough share of wall time per stage, for the progress bar.
STAGE_END = {"prepare": 0.05, "transcribe": 0.40, "diarize": 0.75, "assign": 0.78,
             "names": 0.80, "insights": 1.0}
STAGE_START = {"prepare": 0.0, "transcribe": 0.05, "diarize": 0.40, "assign": 0.75,
               "names": 0.78, "insights": 0.80}


def process(job: dict) -> None:
    meeting_id = str(job["meeting_id"])
    m = db.get_meeting(meeting_id)
    source = Path(m["audio_uris"]["upload"])
    extractor = (job.get("params") or {}).get("extractor")   # notes-only job with another model
    if not extractor:
        db.update_meeting(meeting_id, status="processing", started_at=datetime.now(UTC))
    cfg = PipelineConfig(
        title=m["title"], attendees=m["attendees"] or None,
        num_speakers=m["num_speakers"],
        hotwords=", ".join(m["attendees"]) if m["attendees"] else None,
        insights_extractor=extractor)

    def on_event(e: dict) -> None:
        stage = next((s for s in ("insights", "names") if e["stage"].startswith(s)), e["stage"])
        progress = STAGE_END[stage] if e["status"] in ("done", "cached") else STAGE_START[stage]
        db.update_job(job["id"], stage=stage, progress=progress)
        log.info("job %s %s %s", job["id"], e["stage"], e["status"])

    run(meeting_id, source, cfg, on_event=on_event)
    store = MeetingStore(MEETINGS, meeting_id)
    insights = store.read_json(insights_stage(extractor), "insights.json")
    if extractor:
        # transcript, speakers and the meeting's status are untouched; the run is added beside
        # the existing ones so the UI can compare them
        db.add_insight_run(meeting_id, insights)
    else:
        utterances = store.read_json("assign", "utterances.json")
        duration = store.manifest("prepare")["info"]["duration_s"]
        db.store_results(meeting_id, utterances, insights, duration)
        identify.identify_meeting(meeting_id)   # voiceprints + transcript cues -> speaker names
        db.update_meeting(meeting_id, audio_uris=db.Jsonb(
            {**m["audio_uris"], "prepared": str(store.dir("prepare") / "audio.wav")}))
    db.update_job(job["id"], status="done", progress=1.0, finished_at=datetime.now(UTC))


def main(poll_s: float = 2.0) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    db.apply_schema()
    if n := db.requeue_stale_jobs():
        log.info("requeued %d interrupted job(s)", n)
    log.info("worker ready")
    while True:
        job = db.claim_job()
        if job is None:
            time.sleep(poll_s)
            continue
        log.info("claimed job %s for meeting %s", job["id"], job["meeting_id"])
        try:
            process(job)
        except Exception as exc:  # record and keep serving other jobs
            log.exception("job %s failed", job["id"])
            db.update_job(job["id"], status="failed", finished_at=datetime.now(UTC),
                          error="".join(traceback.format_exception_only(exc)).strip()[:2000])
            if not (job.get("params") or {}).get("extractor"):  # a failed extra run keeps the meeting
                db.update_meeting(str(job["meeting_id"]), status="failed",
                                  error=f"{type(exc).__name__}: {exc}"[:2000])


if __name__ == "__main__":
    main()
