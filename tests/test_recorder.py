"""Desktop recorder through the API, with a fake sound card (no devices needed).
Uses the notetaker_test database like test_api.py; skipped if Postgres is not running."""

import json
import os
import shutil
import time

import numpy as np
import pytest
import soundfile as sf

os.environ["DATABASE_URL"] = "postgresql://notetaker:notetaker@127.0.0.1:5434/notetaker_test"

from services import db

try:
    db.apply_schema()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Postgres not available: {exc}", allow_module_level=True)

from fastapi.testclient import TestClient

from services import api, identify
from services import recorder as recmod
from services import recorder_local
from services.recorder import Device, Recorder

SR = recmod.SR


class FakeReader:
    """Delivers audio in real time, like WASAPI: a tone on the mic, noise on the output.
    `clock` > 1 is a device whose crystal runs fast (more samples than wall time)."""

    def __init__(self, kind, clock=1.0):
        self.kind, self.clock = kind, clock
        self.t0 = None
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.n += numframes
        time.sleep(max(0.0, self.t0 + self.n / (SR * self.clock) - time.monotonic()))
        if self.kind == "mic":
            t = np.arange(numframes) / SR
            return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[:, None]
        return (np.random.default_rng().normal(size=(numframes, 2)) * 0.05).astype(np.float32)


class FakeBackend:
    def __init__(self, mic_clock=1.0):
        self.outs = [Device("spk", "Speakers (fake)", "others")]
        self.mic_clock = mic_clock

    def default_mic(self):
        return Device("mic", "Microphone (fake)", "mic")

    def outputs(self):
        return list(self.outs)

    def open(self, device):
        return FakeReader(device.kind, self.mic_clock if device.kind == "mic" else 1.0)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(recorder_local, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(recmod, "MONITOR_S", 0.2)
    monkeypatch.setattr(recmod, "CALL_END_GRACE_S", 0.6)
    calls = {"now": []}
    r = Recorder(backend=FakeBackend(), root=tmp_path / "recordings",
                 call_detector=lambda: list(calls["now"]))
    monkeypatch.setattr(api, "recorder", r)
    with db.connect() as conn:
        conn.execute("TRUNCATE meetings, persons CASCADE")
    with TestClient(api.app) as c:
        c.calls = calls
        c.rec = r
        yield c


def wait(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_record_stop_creates_a_queued_meeting(client, tmp_path):
    assert client.post("/api/recorder/start", json={"title": "Sync"}).status_code == 422  # no consent
    r = client.post("/api/recorder/start",
                    json={"title": "Sync", "attendees": "Ana, Raj", "your_name": "Rohit",
                          "consent": True})
    assert r.status_code == 201 and r.json()["state"] == "recording"
    assert client.post("/api/recorder/start", json={"consent": True}).status_code == 409

    assert wait(lambda: client.get("/api/recorder").json()["recording"]["levels"]["mic"] > -40)
    s = client.get("/api/recorder").json()["recording"]
    assert s["devices"] == {"mic": "Microphone (fake)", "outputs": ["Speakers (fake)"]}
    assert s["levels"]["others"] > -40 and not s["warnings"]
    assert client.post("/api/recorder/mute", json={"muted": True}).status_code == 204
    assert wait(lambda: client.get("/api/recorder").json()["recording"]["levels"]["mic"] == -100)
    client.post("/api/recorder/mute", json={"muted": False})
    time.sleep(1.0)

    mid = client.post("/api/recorder/stop").json()["meeting_id"]
    m = client.get(f"/api/meetings/{mid}").json()["meeting"]
    assert m["title"] == "Sync" and m["mode"] == "live" and m["owner_name"] == "Rohit"
    assert m["attendees"] == ["Ana", "Raj"]
    assert wait(lambda: db.latest_job(mid) is not None)           # mixed, then queued
    assert wait(lambda: client.get("/api/recorder").json()["state"] == "idle")
    wav, sr = sf.read(tmp_path / "uploads" / mid / "source.wav")
    meta = json.loads((tmp_path / "uploads" / mid / "recording.json").read_text())
    assert sr == SR and abs(len(wav) / SR - meta["elapsed_s"]) < 0.3   # audio length = wall time
    assert (tmp_path / "uploads" / mid / "levels.npz").exists()
    assert not any((tmp_path / "recordings").iterdir())           # raw tracks cleaned up
    assert client.get("/api/recorder").json()["last"]["meeting_id"] == mid


def test_recording_stops_itself_when_the_call_ends(client):
    client.calls["now"] = ["Zoom"]
    client.post("/api/recorder/start", json={"consent": True})
    s = client.get("/api/recorder").json()
    assert s["recording"]["call_app"] == "Zoom" and s["recording"]["title"].startswith("Zoom call, ")
    client.calls["now"] = []                                       # Zoom released the mic
    assert wait(lambda: client.get("/api/recorder").json()["state"] == "idle")
    last = client.get("/api/recorder").json()["last"]
    assert last["reason"] == "The Zoom call ended"


def test_headphones_plugged_in_mid_meeting_are_recorded_too(client):
    client.post("/api/recorder/start", json={"consent": True})
    client.rec.backend.outs.append(Device("hp", "Headphones (fake)", "others"))
    assert wait(lambda: "Headphones (fake)" in
                client.get("/api/recorder").json()["recording"]["devices"]["outputs"])
    assert client.post("/api/recorder/discard").status_code == 204
    assert client.get("/api/recorder").json()["state"] == "idle"
    assert client.get("/api/meetings").json() == []                # nothing kept


def test_interrupted_recording_can_still_become_a_meeting(client, tmp_path):
    client.post("/api/recorder/start", json={"title": "Before the reboot", "consent": True})
    time.sleep(1.5)
    # simulate a crash: copy the live folder as if the process had died mid-recording
    live = client.rec.session.dir
    crashed = tmp_path / "recordings" / "crashed00001"
    shutil.copytree(live, crashed)
    meta = json.loads((crashed / "meta.json").read_text())
    (crashed / "meta.json").write_text(json.dumps({**meta, "id": "crashed00001"}))
    client.post("/api/recorder/discard")

    found = client.get("/api/recorder").json()["interrupted"]
    assert [(r["id"], r["title"]) for r in found] == [("crashed00001", "Before the reboot")]
    assert found[0]["duration_s"] > 0.5
    mid = client.post("/api/recorder/interrupted/crashed00001").json()["meeting_id"]
    assert db.latest_job(mid)["status"] == "queued"
    assert client.get("/api/recorder").json()["interrupted"] == []
    assert client.post("/api/recorder/interrupted/nope").status_code == 404


def test_the_person_recording_is_named_from_the_mic(client, tmp_path, monkeypatch):
    """identify: the cluster whose speech is mic-only gets the owner's name, method 'mic'."""
    monkeypatch.setattr(identify, "MEETINGS", tmp_path / "meetings")
    mid = db.create_recording("Sync", [], "Rohit", recmod.SOURCE_NAME)
    utts = [{"id": "u0000", "speaker": "SPEAKER_00", "start": 0.0, "end": 20.0, "text": "Hi.",
             "words": []},
            {"id": "u0001", "speaker": "SPEAKER_01", "start": 20.0, "end": 40.0, "text": "Hey.",
             "words": []}]
    ins = {"meta": {"extractor": "claude-opus-5-5", "prompt_version": 2, "schema_version": 1,
                    "status": {}}, "summary": "", "topics": [], "items": []}
    db.store_results(mid, utts, ins, 40.0)
    d = tmp_path / "meetings" / mid / "diarize"
    d.mkdir(parents=True)
    turns = [{"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
             {"start": 20.0, "end": 40.0, "speaker": "SPEAKER_01"}]
    (d / "diarization.json").write_text(json.dumps({"regular": turns, "exclusive": turns}))
    n = 40 * SR // 320
    mic_db = np.full(n, -80.0, np.float32)
    oth_db = np.full(n, -100.0, np.float32)
    oth_db[: n // 2] = -25.0          # SPEAKER_00 came through the loopback
    mic_db[n // 2:] = -25.0           # SPEAKER_01 only on the mic: the owner
    levels = tmp_path / "levels.npz"
    np.savez(levels, mic_db=mic_db, oth_db=oth_db, lag=0)
    db.update_meeting(mid, audio_uris=db.Jsonb({"levels": str(levels)}))

    assert identify.identify_meeting(mid)["mic"] == 1
    spk = {s["cluster_label"]: s for s in client.get(f"/api/meetings/{mid}").json()["speakers"]}
    assert spk["SPEAKER_01"]["display_name"] == "Rohit" and spk["SPEAKER_01"]["method"] == "mic"
    assert spk["SPEAKER_00"]["display_name"] is None


def test_a_fast_device_clock_is_locked_to_wall_time(tmp_path):
    """A mic whose clock runs 0.2 % fast (4x the ~500 ppm measured on this laptop)
    stays locked to the system clock, so it lines up with the loopback track."""
    r = Recorder(backend=FakeBackend(mic_clock=1.002), root=tmp_path, call_detector=list)
    r.start("drift", [], None)
    time.sleep(8)
    s = r.session
    mic = next(w for w in s.writers if w.device.kind == "mic")
    out = next(w for w in s.writers if w.device.kind == "others")
    s.halt()
    # unlocked, the mic would be ~(8 s - 2 s baseline) * 0.2 % = 192 samples ahead by now
    assert abs(mic.drift) < 80 and mic.adjusted > 120     # within 5 ms (vs 12 ms unlocked)
    assert abs(out.drift) < 80           # a steady clock is not pushed off (jitter corrections cancel)
