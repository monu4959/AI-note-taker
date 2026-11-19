"""A teammate's recorder agent delivering to the notetaker server (both in-process: the
agent's RemoteSink talks to the real API app through a TestClient). Uses notetaker_test."""

import os
import time

import pytest
import soundfile as sf

os.environ["DATABASE_URL"] = "postgresql://notetaker:notetaker@127.0.0.1:5434/notetaker_test"

from services import db

try:
    db.apply_schema()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Postgres not available: {exc}", allow_module_level=True)

import httpx
from fastapi.testclient import TestClient
from test_recorder import FakeBackend, wait

from services import api, recorder_local
from services.agent import RemoteSink
from services.recorder import Recorder


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(recorder_local, "UPLOADS", tmp_path / "uploads")
    with db.connect() as conn:
        conn.execute("TRUNCATE meetings, persons CASCADE")
    with TestClient(api.app) as c:
        yield c


def test_agent_recording_arrives_on_the_server_as_a_queued_meeting(server, tmp_path):
    r = Recorder(backend=FakeBackend(), root=tmp_path / "agent", call_detector=list,
                 sink=RemoteSink(server="", client=server))
    r.start("Standup on a Mac", ["Ana"], "Raj")
    time.sleep(1.5)
    mid = r.stop("Stopped by you")
    assert mid
    m = server.get(f"/api/meetings/{mid}").json()["meeting"]
    assert m["mode"] == "live" and m["owner_name"] == "Raj" and m["attendees"] == ["Ana"]
    assert wait(lambda: db.latest_job(mid) is not None)            # audio arrived, job queued
    assert wait(lambda: r.state == "idle")
    src = tmp_path / "uploads" / mid / "source.flac"
    info = sf.info(src)
    assert info.format == "FLAC" and info.samplerate == 16000 and info.duration > 1.0
    assert (tmp_path / "uploads" / mid / "levels.npz").exists()
    assert r.interrupted() == []                                    # local copy removed
    # the same audio can't be uploaded twice
    with src.open("rb") as fh:
        again = server.post(f"/api/recordings/{mid}/audio",
                            files={"source": ("s.flac", fh), "levels": ("l.npz", b"x")})
    assert again.status_code == 409


def test_server_unreachable_keeps_the_audio_and_it_can_be_sent_later(server, tmp_path):
    down = RemoteSink(server="http://127.0.0.1:9", client=httpx.Client(timeout=2))
    r = Recorder(backend=FakeBackend(), root=tmp_path / "agent", call_detector=list, sink=down)
    r.start("Offline call", [], None)
    time.sleep(1.2)
    assert r.stop("Stopped by you") is None
    assert "Couldn't reach the notetaker server" in r.status()["last"]["error"]
    kept = r.interrupted()
    assert [k["title"] for k in kept] == ["Offline call"]

    r._sink = RemoteSink(server="", client=server)                  # the server is back
    mid = r.recover(kept[0]["id"])
    assert db.latest_job(mid)["status"] == "queued" and r.interrupted() == []
