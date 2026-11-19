"""REST API (design doc "Endpoints"): uploads, the desktop recorder (Step 2), recordings
from teammates' recorder agents, meetings, speakers and notes. Also serves the built web app
(apps/web/dist) so teammates only need a browser.

Usage: .venv/Scripts/python -m uvicorn services.api:app --port 8000
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import llm
from pipeline.run import MEETINGS
from services import db, identify, recorder_api, recorder_local, security
from services.recorder import Recorder

log = logging.getLogger("api")

UPLOADS = db.ROOT / "data" / "uploads"
ALLOWED = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac", ".aac", ".mkv", ".mov"}
MAX_BYTES = 2 * 1024**3
MAX_LEVELS = 64 * 1024**2


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.apply_schema()
    yield


app = FastAPI(title="Meeting notetaker", lifespan=lifespan)
recorder = Recorder()   # the one desktop recorder of this machine (opens devices lazily)


def this_machine_only(request: Request) -> None:
    """The server's own recorder records THIS computer: only a browser on it may use it.
    Teammates record with the recorder agent on their own computer."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1") and not (security.testing() and host == "testclient"):
        raise HTTPException(403, "recording runs on your own computer: use the recorder app")


app.include_router(recorder_api.router(lambda: recorder),
                   dependencies=[Depends(this_machine_only)])

# No sign-in yet: network, Host, Origin and upload-size checks keep the pilot private
# (services/security.py).
allowed = security.client_allowed


@app.middleware("http")
async def private_pilot_checks(request: Request, call_next):
    refused = security.check(request)
    return refused if refused is not None else await call_next(request)


def _meeting_or_404(meeting_id: str) -> dict:
    try:
        m = db.get_meeting(meeting_id)
    except Exception:  # malformed UUID
        m = None
    if m is None:
        raise HTTPException(404, "meeting not found")
    return m


@app.get("/api/health")
def health():
    with db.connect() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


@app.post("/api/meetings/upload", status_code=201)
async def upload(file: UploadFile = File(...), title: str = Form(""),
                 attendees: str = Form(""), num_speakers: int | None = Form(None)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"unsupported file type {suffix!r}")
    people = [a.strip() for a in attendees.split(",") if a.strip()]
    meeting_id = db.create_meeting(title.strip() or Path(file.filename).stem, people,
                                   num_speakers, file.filename)
    dest = UPLOADS / meeting_id / ("source" + suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                fh.close()
                shutil.rmtree(dest.parent, ignore_errors=True)
                db.delete_meeting(meeting_id)
                raise HTTPException(413, "file too large (2 GB max)")
            fh.write(chunk)
    db.set_upload(meeting_id, dest)  # also enqueues the job
    return {"id": meeting_id}


# --- recordings from teammates' recorder agents (services/agent.py) ---------------------

class NewRecording(BaseModel):
    title: str
    attendees: list[str] = []
    owner_name: str | None = None


@app.post("/api/recordings", status_code=201)
def create_recording(body: NewRecording):
    """A teammate's recorder stopped: create the meeting (its audio follows)."""
    if body.owner_name and not recorder_api.NAME.fullmatch(body.owner_name):
        raise HTTPException(422, "owner_name: letters, spaces, . ' - only (max 60)")
    return {"id": recorder_local.create(body.model_dump())}


@app.post("/api/recordings/{meeting_id}/audio", status_code=202)
async def recording_audio(meeting_id: str, source: UploadFile = File(...),
                          levels: UploadFile = File(...), meta: str = Form("{}")):
    """The mixed recording (FLAC) and its levels; queues the normal pipeline job."""
    m = _meeting_or_404(meeting_id)
    if m["mode"] != "live" or m["audio_uris"].get("upload"):
        raise HTTPException(409, "this meeting already has its audio")
    up = UPLOADS / meeting_id
    up.mkdir(parents=True, exist_ok=True)
    src, lv = up / "source.flac", up / "levels.npz"
    size = 0
    with src.open("wb") as fh:
        while chunk := await source.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                fh.close()
                shutil.rmtree(up, ignore_errors=True)
                raise HTTPException(413, "recording too large (2 GB max)")
            fh.write(chunk)
    data = await levels.read(MAX_LEVELS + 1)     # per-frame levels: ~1 MB per hour
    if len(data) > MAX_LEVELS:
        shutil.rmtree(up, ignore_errors=True)
        raise HTTPException(413, "levels file too large")
    lv.write_bytes(data)
    try:
        info = json.loads(meta)
    except json.JSONDecodeError:
        info = {}
    recorder_local.accept(meeting_id, src, lv, info)
    return {"id": meeting_id}


class Failure(BaseModel):
    error: str


@app.post("/api/recordings/{meeting_id}/failed", status_code=204)
def recording_failed(meeting_id: str, body: Failure):
    m = _meeting_or_404(meeting_id)
    if not m["audio_uris"].get("upload"):
        db.update_meeting(meeting_id, status="failed", error=body.error[:2000])


@app.get("/api/meetings")
def meetings():
    return db.list_meetings()


@app.get("/api/meetings/{meeting_id}")
def meeting(meeting_id: str):
    m = _meeting_or_404(meeting_id)
    return {
        "meeting": {k: v for k, v in m.items() if k != "audio_uris"},
        "job": db.latest_job(meeting_id),
        "speakers": _with_samples(db.speakers(meeting_id), utts := db.transcript(meeting_id)),
        "transcript": utts,
    }


def _with_samples(speakers: list[dict], utts: list[dict], max_ms: int = 8000) -> list[dict]:
    """Attach a short sample of each speaker's voice: the start of their longest utterance."""
    longest: dict[str, dict] = {}
    for u in utts:
        cur = longest.get(u["speaker_label"])
        if cur is None or u["end_ms"] - u["start_ms"] > cur["end_ms"] - cur["start_ms"]:
            longest[u["speaker_label"]] = u
    for s in speakers:
        u = longest.get(s["cluster_label"])
        s["sample"] = ({"start_ms": u["start_ms"], "end_ms": min(u["end_ms"], u["start_ms"] + max_ms)}
                       if u else None)
        s["speech_ms"] = sum(x["end_ms"] - x["start_ms"] for x in utts
                             if x["speaker_label"] == s["cluster_label"])
    return speakers


# Models that can write notes with the active provider (pipeline/llm.py, NOTES_LLM in .env).
# With NOTES_LLM=local only the local model is offered: Claude can't be requested at all.
NOTE_MODELS = ({"opus": llm.CLAUDE_EXTRACTOR, "sonnet": llm.CLAUDE_VERIFIER}
               if llm.PROVIDER == "claude" else {"local": llm.LOCAL_MODEL})


@app.get("/api/config")
def config():
    """What the web app needs to know about this server: which notes provider is active."""
    return {"notes_llm": llm.PROVIDER, "extractor": llm.EXTRACTOR, "verifier": llm.VERIFIER,
            "note_models": NOTE_MODELS}


@app.get("/api/meetings/{meeting_id}/insight-runs")
def insight_runs(meeting_id: str):
    """Notes per model (newest run of each), to switch between and compare."""
    _meeting_or_404(meeting_id)
    return db.insight_runs(meeting_id)


class NotesRequest(BaseModel):
    model: str


@app.post("/api/meetings/{meeting_id}/notes", status_code=202)
def write_notes(meeting_id: str, body: NotesRequest):
    """Write another set of notes with a different model. Transcription, speakers and names
    are reused, so only the LLM steps run (about $0.30 per meeting hour with Sonnet)."""
    m = _meeting_or_404(meeting_id)
    if body.model not in NOTE_MODELS:
        raise HTTPException(422, f"model must be one of {sorted(NOTE_MODELS)}")
    if m["status"] != "ready":
        raise HTTPException(409, "the meeting is still processing")
    job = db.latest_job(meeting_id)
    if job and job["status"] in ("queued", "running"):
        raise HTTPException(409, "notes are already being written for this meeting")
    return db.enqueue_notes(meeting_id, NOTE_MODELS[body.model])


@app.get("/api/meetings/{meeting_id}/insights")
def insights(meeting_id: str, include_rejected: bool = False, run_id: int | None = None):
    _meeting_or_404(meeting_id)
    run = db.latest_insights(meeting_id, run_id)
    if run is None:
        raise HTTPException(404, "no insights yet")
    payload = run["payload"]
    items = [it for it in payload["items"]
             if include_rejected or it["verification"]["status"] != "rejected"]
    return {"run_id": run["id"], "created_at": run["created_at"], "meta": payload["meta"],
            "summary": payload["summary"], "topics": payload["topics"], "items": items}


class AskTurn(BaseModel):
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[AskTurn] = Field(default=[], max_length=20)


@app.post("/api/meetings/{meeting_id}/ask")
def ask_meeting(meeting_id: str, body: AskRequest):
    """Answer a question about the meeting from its transcript, with cited lines
    (pipeline/ask.py). Claude: about $0.03 for the first question, less after; local: free."""
    from pipeline import ask, llm_local

    m = _meeting_or_404(meeting_id)
    rows = db.transcript(meeting_id)
    if not rows:
        raise HTTPException(409, "this meeting has no transcript yet")
    run = db.latest_insights(meeting_id, None)
    try:
        return ask.ask(meeting_id, m["title"], rows, db.speakers(meeting_id),
                       run["payload"] if run else None, body.question.strip(),
                       [t.model_dump() for t in body.history])
    except llm.BudgetExceeded as e:
        raise HTTPException(402, f"The LLM budget is used up ({e}). Raise LLM_BUDGET_USD in .env.")
    except llm_local.LocalLLMError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001  (Claude unreachable, refusal, ...): say so, don't 500
        log.exception("ask failed")
        raise HTTPException(502, f"Couldn't answer right now: {type(e).__name__}")


@app.get("/api/meetings/{meeting_id}/audio")
def audio(meeting_id: str):
    """Prepared 16 kHz wav; FileResponse honours Range requests, so the player can seek."""
    m = _meeting_or_404(meeting_id)
    path = m["audio_uris"].get("prepared")
    if not path or not Path(path).exists():
        raise HTTPException(404, "audio not ready")
    return FileResponse(path, media_type="audio/wav")


class SpeakerName(BaseModel):
    display_name: str | None
    remember_voice: bool = False   # opt-in: store this speaker's voiceprint under the name


@app.patch("/api/meetings/{meeting_id}/speakers/{label}")
def rename(meeting_id: str, label: str, body: SpeakerName):
    """Confirm or rename a speaker (empty name = back to anonymous); optionally remember the
    voice so later meetings recognise it."""
    _meeting_or_404(meeting_id)
    name = (body.display_name or "").strip() or None
    if name and not re.fullmatch(r"[\w .'\-]{1,60}", name):
        raise HTTPException(422, "name: letters, spaces, . ' - only (max 60)")
    try:
        return identify.confirm(meeting_id, label, name, body.remember_voice)
    except KeyError:
        raise HTTPException(404, "speaker not found") from None
    except identify.EnrollError as e:
        raise HTTPException(422, str(e)) from None


@app.post("/api/meetings/{meeting_id}/identify")
def reidentify(meeting_id: str):
    """Re-run automatic naming (voiceprints + transcript cues) for this meeting."""
    _meeting_or_404(meeting_id)
    return identify.identify_meeting(meeting_id)


@app.get("/api/people")
def people():
    """People whose voice is remembered (voiceprints are opt-in and deletable)."""
    return identify.people()


@app.delete("/api/people/{person_id}", status_code=204)
def forget_person(person_id: int):
    """Forget a person: deletes every stored voiceprint and their automatic names."""
    if not identify.forget(person_id):
        raise HTTPException(404, "person not found")


@app.delete("/api/meetings/{meeting_id}", status_code=204)
def delete(meeting_id: str):
    """Deletes the meeting's rows, the uploaded file and all pipeline artifacts."""
    _meeting_or_404(meeting_id)
    db.delete_meeting(meeting_id)
    shutil.rmtree(UPLOADS / meeting_id, ignore_errors=True)
    shutil.rmtree(MEETINGS / meeting_id, ignore_errors=True)


# The built web app (npm run build), served last so /api/* routes win.
WEB = db.ROOT / "apps" / "web" / "dist"
if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
