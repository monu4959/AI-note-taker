"""HTTP endpoints of the desktop recorder, shared by the notetaker server (services/api.py)
and the recorder agent that runs on a teammate's computer (services/agent.py)."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.recorder import Recorder


class RecordingStart(BaseModel):
    title: str = ""
    attendees: str = ""
    your_name: str = ""
    consent: bool = False   # "everyone in this meeting knows it is being recorded"


class Mute(BaseModel):
    muted: bool


NAME = re.compile(r"[\w .'\-]{1,60}")


def router(get: Callable[[], Recorder]) -> APIRouter:
    r = APIRouter(prefix="/api/recorder")

    def call(fn, *args):
        try:
            return fn(*args)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        except KeyError:
            raise HTTPException(404, "recording not found") from None

    @r.get("")
    def status():
        return get().status()

    @r.get("/events")
    async def events(request: Request):
        """Server-sent events: the recorder status, 4x a second while recording (levels,
        timer), every 1.5 s otherwise (call detection). One request instead of polling; the
        stream ends after 5 minutes and the browser's EventSource reconnects by itself."""
        async def stream():
            last, sent, opened = None, 0.0, time.monotonic()
            yield "retry: 1000\n\n"
            while time.monotonic() - opened < 300 and not await request.is_disconnected():
                s = await asyncio.to_thread(get().status)
                data = json.dumps(s, default=str)
                if data != last or time.monotonic() - sent > 15:   # 15 s heartbeat
                    yield f"data: {data}\n\n"
                    last, sent = data, time.monotonic()
                await asyncio.sleep(0.25 if s["state"] == "recording" else 1.5)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @r.post("/start", status_code=201)
    def start(body: RecordingStart):
        if not body.consent:
            raise HTTPException(422, "confirm that everyone in the meeting knows it is being recorded")
        name = body.your_name.strip() or None
        if name and not NAME.fullmatch(name):
            raise HTTPException(422, "your name: letters, spaces, . ' - only (max 60)")
        people = [a.strip() for a in body.attendees.split(",") if a.strip()]
        rec = get()
        app = (rec.active_calls() or [None])[0]
        title = body.title.strip() or (
            f"{app + ' call' if app else 'Meeting'}, "
            f"{time.strftime('%b %d, %I:%M %p').replace(' 0', ' ')}")
        return call(rec.start, title, people, name)

    @r.post("/stop")
    def stop():
        """Stops and returns the new meeting at once (null if the server can't be reached;
        the audio is then kept); its audio is saved and queued shortly after."""
        return {"meeting_id": call(get().stop, "Stopped by you")}

    @r.post("/mute", status_code=204)
    def mute(body: Mute):
        """Pause your microphone in the recording (the others keep being recorded)."""
        call(get().set_muted, body.muted)

    @r.post("/discard", status_code=204)
    def discard():
        """Stop and delete the recording in progress; nothing is kept."""
        call(get().discard)

    @r.post("/interrupted/{rec_id}")
    def recover(rec_id: str):
        """Turn a recording that was cut short (crash, reboot, server unreachable) into a
        meeting."""
        try:
            return {"meeting_id": call(get().recover, rec_id)}
        except HTTPException:
            raise
        except Exception as e:   # e.g. still can't reach the server
            raise HTTPException(502, f"couldn't send the recording: {e}") from None

    @r.delete("/interrupted/{rec_id}", status_code=204)
    def drop(rec_id: str):
        call(get().drop, rec_id)

    return r
