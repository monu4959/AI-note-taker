"""The recorder agent: what a teammate runs on their own Mac or PC.

It records calls on that computer (services/recorder.py, the same code as on the server)
and sends each finished recording to the team's notetaker server. It also serves the web
app at http://localhost:8765 and forwards every other /api request to the server, so the
teammate gets the whole app (meetings, notes, speakers) plus the Record button, from one
address, with no database, GPU or API keys on their machine.

    browser ── http://localhost:8765 ──► agent ──► /api/recorder/*  recorded here
                                                └─► /api/*          forwarded to the server

Usage:  NOTETAKER_SERVER=http://<server>:8000 python -m services.agent
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from services import recorder_api, security
from services.recorder import ROOT, Recorder, mix_recording

log = logging.getLogger("agent")

SERVER = os.environ.get("NOTETAKER_SERVER", "http://127.0.0.1:8000").strip().rstrip("/")
PORT = int(os.environ.get("NOTETAKER_AGENT_PORT", "8765"))
WEB = Path(os.environ.get("NOTETAKER_WEB", ROOT / "apps" / "web" / "dist"))


def to_flac(wav: Path, flac: Path) -> None:
    """Lossless, about half the size of the wav (~55 MB per hour), streamed in blocks."""
    with sf.SoundFile(wav) as src, sf.SoundFile(flac, "w", src.samplerate, 1,
                                                format="FLAC", subtype="PCM_16") as dst:
        for block in src.blocks(blocksize=src.samplerate * 60, dtype="int16"):
            dst.write(np.asarray(block))


class RemoteSink:
    """Deliver recordings to the notetaker server over HTTP."""

    def __init__(self, server: str = SERVER, client: httpx.Client | None = None):
        self.server = server
        self.http = client or httpx.Client()

    def create(self, meta: dict) -> str:
        r = self.http.post(f"{self.server}/api/recordings", timeout=15, json={
            "title": meta["title"], "attendees": meta.get("attendees") or [],
            "owner_name": meta.get("owner_name")})
        r.raise_for_status()
        return r.json()["id"]

    def exists(self, meeting_id: str) -> bool:
        r = self.http.get(f"{self.server}/api/meetings/{meeting_id}", timeout=15)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    def deliver(self, rec_dir: Path, meeting_id: str, meta: dict) -> dict:
        out = rec_dir / "out"
        out.mkdir(exist_ok=True)
        info = mix_recording(rec_dir, out / "source.wav", out / "levels.npz")
        to_flac(out / "source.wav", out / "source.flac")
        with (out / "source.flac").open("rb") as src, (out / "levels.npz").open("rb") as lv:
            r = self.http.post(f"{self.server}/api/recordings/{meeting_id}/audio", timeout=900,
                           files={"source": ("source.flac", src, "audio/flac"),
                                  "levels": ("levels.npz", lv, "application/octet-stream")},
                           data={"meta": json.dumps({**meta, "mix": info}, default=str)})
        r.raise_for_status()
        shutil.rmtree(rec_dir, ignore_errors=True)   # only once the server has it
        return info

    def fail(self, meeting_id: str, error: str) -> None:
        self.http.post(f"{self.server}/api/recordings/{meeting_id}/failed", timeout=15,
                       json={"error": error})


app = FastAPI(title="Notetaker recorder")
recorder = Recorder(sink=RemoteSink())


@app.middleware("http")
async def private_checks(request: Request, call_next):
    """Same checks as the server: a web page on another site can't reach this recorder or,
    through it, the team's meetings (services/security.py)."""
    refused = security.check(request)
    return refused if refused is not None else await call_next(request)
app.include_router(recorder_api.router(lambda: recorder))

HOP = {"host", "connection", "keep-alive", "transfer-encoding", "te", "upgrade",
       "proxy-authorization", "proxy-connection", "content-length"}
client = httpx.AsyncClient(base_url=SERVER, timeout=httpx.Timeout(30, read=900))


@app.get("/api/agent")
def agent_info():
    """Lets the web app know it is talking to a recorder agent, and which server."""
    return {"agent": True, "server": SERVER}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def forward(path: str, request: Request):
    """Everything that isn't recording goes to the server, streamed both ways (uploads,
    audio with Range requests for seeking)."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    req = client.build_request(request.method, f"/api/{path}", params=request.query_params,
                               headers=headers, content=request.stream())
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        return JSONResponse({"detail": f"the notetaker server ({SERVER}) can't be reached: {e}"},
                            status_code=502)
    keep = {k: v for k, v in resp.headers.items()
            if k.lower() in {"content-type", "content-length", "content-range", "accept-ranges",
                             "cache-control", "content-disposition", "etag", "last-modified",
                             "location"}}
    return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=keep,
                             background=BackgroundTask(resp.aclose))


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("recorder agent on http://localhost:%d, server %s", PORT, SERVER)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
