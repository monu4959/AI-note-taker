"""Where recordings go on the notetaker server: its database, uploads folder and GPU worker.

Used by the recorder running on the server machine itself (LocalSink), and by the API when a
teammate's recorder agent uploads a finished recording (`accept`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from services import db
from services.recorder import SOURCE_NAME, mix_recording

UPLOADS = db.ROOT / "data" / "uploads"


def create(meta: dict) -> str:
    return db.create_recording(meta["title"], meta.get("attendees") or [],
                               meta.get("owner_name"), SOURCE_NAME)


def accept(meeting_id: str, src: Path, levels: Path, meta: dict) -> None:
    """A mixed recording is in the meeting's upload folder: record it and queue the job."""
    (src.parent / "recording.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    m = db.get_meeting(meeting_id)
    db.update_meeting(meeting_id, audio_uris=db.Jsonb(
        {**(m["audio_uris"] if m else {}), "levels": str(levels)}))
    db.set_upload(meeting_id, src)                 # queues the transcribe -> notes job


class LocalSink:
    def create(self, meta: dict) -> str:
        return create(meta)

    def exists(self, meeting_id: str) -> bool:
        return db.get_meeting(meeting_id) is not None

    def deliver(self, rec_dir: Path, meeting_id: str, meta: dict) -> dict:
        up = UPLOADS / meeting_id
        src, levels = up / "source.wav", up / "levels.npz"
        info = mix_recording(rec_dir, src, levels)
        accept(meeting_id, src, levels, {**meta, "mix": info})
        shutil.rmtree(rec_dir, ignore_errors=True)
        return info

    def fail(self, meeting_id: str, error: str) -> None:
        db.update_meeting(meeting_id, status="failed", error=error[:2000])
