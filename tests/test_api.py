"""API + job queue integration test against the real Postgres (separate notetaker_test DB).
Skipped if the database container is not running."""

import io
import os

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

from services import api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(api, "MEETINGS", tmp_path / "meetings")
    with db.connect() as conn:
        conn.execute("TRUNCATE meetings, persons CASCADE")
    with TestClient(api.app) as c:
        yield c


def _wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), np.float32), 16000, format="WAV")
    return buf.getvalue()


UTTS = [
    {"id": "u0000", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "We ship Friday.",
     "words": [{"word": " We", "start": 0.0, "end": 0.2, "prob": 0.9}]},
    {"id": "u0001", "speaker": "SPEAKER_01", "start": 1.2, "end": 2.0, "text": "Agreed.",
     "words": []},
]
INSIGHTS = {
    "meta": {"extractor": "claude-opus-5-5", "prompt_version": 2, "schema_version": 1,
             "status": {"verified": 1, "needs_review": 0, "rejected": 1}},
    "summary": "They agreed to ship Friday.",
    "topics": [{"id": "t1", "title": "Release", "start_id": "u0000", "end_id": "u0001"}],
    "items": [
        {"id": "d1", "type": "decision", "topic_id": "t1", "text": "Ship on Friday",
         "evidence": {"utterance_ids": ["u0000", "u0001"]},
         "verification": {"status": "verified", "checks": [], "support": "supported"}},
        {"id": "d2", "type": "decision", "topic_id": "t1", "text": "Ship on Monday",
         "evidence": {"utterance_ids": ["u0000"]},
         "verification": {"status": "rejected", "checks": [], "support": "unsupported"}},
    ],
}


def test_upload_queue_results_rename_delete(client, tmp_path):
    r = client.post("/api/meetings/upload", data={"title": "Standup", "attendees": "Priya, Raj"},
                    files={"file": ("call.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 201
    mid = r.json()["id"]
    assert (tmp_path / "uploads" / mid / "source.wav").exists()

    job = db.claim_job()
    assert job and str(job["meeting_id"]) == mid
    assert db.claim_job() is None  # nothing else queued; a claimed job is not handed out twice

    db.store_results(mid, UTTS, INSIGHTS, duration_s=2.0)
    m = client.get(f"/api/meetings/{mid}").json()
    assert m["meeting"]["status"] == "ready" and m["meeting"]["attendees"] == ["Priya", "Raj"]
    assert [u["seq"] for u in m["transcript"]] == [0, 1]
    assert {s["cluster_label"] for s in m["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}

    ins = client.get(f"/api/meetings/{mid}/insights").json()
    assert [i["id"] for i in ins["items"]] == ["d1"]  # rejected hidden by default
    ins_all = client.get(f"/api/meetings/{mid}/insights?include_rejected=true").json()
    assert len(ins_all["items"]) == 2

    r = client.patch(f"/api/meetings/{mid}/speakers/SPEAKER_00", json={"display_name": "Priya"})
    assert r.status_code == 200
    db.store_results(mid, UTTS, INSIGHTS, duration_s=2.0)  # reprocessing keeps the user's name
    names = {s["cluster_label"]: s["display_name"] for s in client.get(f"/api/meetings/{mid}").json()["speakers"]}
    assert names["SPEAKER_00"] == "Priya"

    assert client.get("/api/meetings").json()[0]["id"] == mid
    assert client.delete(f"/api/meetings/{mid}").status_code == 204
    assert client.get(f"/api/meetings/{mid}").status_code == 404
    assert not (tmp_path / "uploads" / mid).exists()


def test_rejects_unsupported_type_and_bad_ids(client):
    r = client.post("/api/meetings/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    assert client.get("/api/meetings/not-a-uuid").status_code == 404


def _fake_diarization(root, meeting_id, embeddings, speech_s=30.0):
    """Write the diarize-stage files identification reads (sorted labels <-> embedding rows)."""
    import json

    d = root / meeting_id / "diarize"
    d.mkdir(parents=True)
    labels = [f"SPEAKER_{i:02d}" for i in range(len(embeddings))]
    turns = [{"start": i * 100.0, "end": i * 100.0 + speech_s, "speaker": lab}
             for i, lab in enumerate(labels)]
    (d / "diarization.json").write_text(json.dumps({"regular": turns, "exclusive": turns}))
    np.save(d / "embeddings.npy", np.array(embeddings, dtype=np.float32))


def test_remember_voice_names_the_next_meeting_and_forget_removes_it(client, tmp_path, monkeypatch):
    from services import identify

    monkeypatch.setattr(identify, "MEETINGS", tmp_path / "meetings")
    rng = np.random.default_rng(0)
    ana, raj = rng.normal(size=256), rng.normal(size=256)

    ids = []
    for _ in range(2):
        ids.append(client.post("/api/meetings/upload", files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
                               data={"title": "Sync"}).json()["id"])
        db.store_results(ids[-1], UTTS, INSIGHTS, 2.0)
    # meeting 1: SPEAKER_00 = Ana, SPEAKER_01 = Raj; meeting 2 has them in the other order
    _fake_diarization(tmp_path / "meetings", ids[0], [ana, raj])
    _fake_diarization(tmp_path / "meetings", ids[1], [raj + rng.normal(scale=0.3, size=256),
                                                      ana + rng.normal(scale=0.3, size=256)])

    r = client.patch(f"/api/meetings/{ids[0]}/speakers/SPEAKER_00",
                     json={"display_name": "Ana", "remember_voice": True})
    assert r.status_code == 200 and r.json()["remembered"]
    people = client.get("/api/people").json()
    assert [(p["display_name"], p["voiceprints"]) for p in people] == [("Ana", 1)]

    spk = {s["cluster_label"]: s for s in client.get(f"/api/meetings/{ids[1]}").json()["speakers"]}
    assert spk["SPEAKER_01"]["display_name"] == "Ana" and spk["SPEAKER_01"]["method"] == "voiceprint"
    assert not spk["SPEAKER_01"]["confirmed"]
    assert spk["SPEAKER_00"]["display_name"] is None          # Raj was never remembered
    assert spk["SPEAKER_01"]["sample"] is not None

    # a name you type is never overwritten; forgetting removes automatic names only
    client.patch(f"/api/meetings/{ids[1]}/speakers/SPEAKER_00", json={"display_name": "Raj"})
    assert client.delete(f"/api/people/{people[0]['id']}").status_code == 204
    spk = {s["cluster_label"]: s for s in client.get(f"/api/meetings/{ids[1]}").json()["speakers"]}
    assert spk["SPEAKER_01"]["display_name"] is None
    assert spk["SPEAKER_00"]["display_name"] == "Raj" and spk["SPEAKER_00"]["confirmed"]
    assert client.get("/api/people").json() == []


def test_remember_voice_needs_enough_speech(client, tmp_path, monkeypatch):
    from services import identify

    monkeypatch.setattr(identify, "MEETINGS", tmp_path / "meetings")
    mid = client.post("/api/meetings/upload", files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
                      data={"title": "Sync"}).json()["id"]
    db.store_results(mid, UTTS, INSIGHTS, 2.0)
    _fake_diarization(tmp_path / "meetings", mid, [np.ones(256), -np.ones(256)], speech_s=3.0)
    r = client.patch(f"/api/meetings/{mid}/speakers/SPEAKER_00",
                     json={"display_name": "Ana", "remember_voice": True})
    assert r.status_code == 422 and "not enough clear speech" in r.json()["detail"]


def test_second_notes_model_is_queued_and_listed_beside_the_first(client):
    """A second set of notes from the other allowed model (Claude mode: Sonnet; local mode:
    the local model), listed beside the first."""
    key, model = next((k, m) for k, m in api.NOTE_MODELS.items() if m != "claude-opus-5-5")
    mid = client.post("/api/meetings/upload", files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
                      data={"title": "Sync"}).json()["id"]
    assert client.post(f"/api/meetings/{mid}/notes", json={"model": key}).status_code == 409  # not ready
    db.store_results(mid, UTTS, INSIGHTS, 2.0)
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET status = 'done' WHERE meeting_id = %s", (mid,))
    assert client.post(f"/api/meetings/{mid}/notes", json={"model": "gpt"}).status_code == 422
    job = client.post(f"/api/meetings/{mid}/notes", json={"model": key}).json()
    assert job["params"] == {"extractor": model} and job["status"] == "queued"
    assert client.post(f"/api/meetings/{mid}/notes", json={"model": key}).status_code == 409  # busy

    second = {**INSIGHTS, "meta": {**INSIGHTS["meta"], "extractor": model, "cost_usd": 0.0},
              "summary": "Second summary."}
    run_id = db.add_insight_run(mid, second)
    runs = {r["model"]: r for r in client.get(f"/api/meetings/{mid}/insight-runs").json()}
    assert set(runs) == {"claude-opus-5-5", model}
    assert runs[model]["id"] == run_id
    opus = runs["claude-opus-5-5"]["id"]
    assert client.get(f"/api/meetings/{mid}/insights?run_id={opus}").json()["summary"] == INSIGHTS["summary"]
    assert client.get(f"/api/meetings/{mid}/insights").json()["summary"] == "Second summary."
    assert client.get(f"/api/meetings/{mid}").json()["meeting"]["status"] == "ready"


def test_claude_cannot_be_requested_while_it_is_switched_off(client):
    from pipeline import llm

    if llm.PROVIDER == "claude":
        pytest.skip("NOTES_LLM=claude on this machine")
    assert client.get("/api/config").json()["notes_llm"] == "local"
    mid = client.post("/api/meetings/upload", files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
                      data={"title": "Sync"}).json()["id"]
    db.store_results(mid, UTTS, INSIGHTS, 2.0)
    with db.connect() as conn:
        conn.execute("UPDATE jobs SET status = 'done' WHERE meeting_id = %s", (mid,))
    for claude in ("opus", "sonnet"):
        assert client.post(f"/api/meetings/{mid}/notes", json={"model": claude}).status_code == 422


def test_only_this_machine_and_tailscale_may_connect():
    assert api.allowed("127.0.0.1") and api.allowed("::1")
    assert api.allowed("100.101.102.103")                 # Tailscale (CGNAT range)
    assert api.allowed("fd7a:115c:a1e0::1234")            # Tailscale IPv6
    assert not api.allowed("192.168.1.20")                # the café Wi-Fi
    assert not api.allowed("8.8.8.8") and not api.allowed("")


def test_pages_on_other_sites_and_rebound_names_are_refused():
    from starlette.requests import Request

    from services import security

    def req(method="GET", host="localhost:8000", **headers):
        hs = [(b"host", host.encode())] + [(k.replace("_", "-").encode(), v.encode())
                                           for k, v in headers.items()]
        return Request({"type": "http", "method": method, "path": "/api/recorder/stop",
                        "headers": hs, "client": ("127.0.0.1", 5000), "query_string": b""})

    assert security.check(req()) is None
    assert security.check(req(host="my-pc.tail1234.ts.net:8000")) is None
    assert security.check(req(host="100.101.102.103:8000")) is None
    assert security.check(req(host="evil.example")).status_code == 403          # DNS rebinding
    assert security.check(req("POST", origin="https://evil.example")).status_code == 403
    assert security.check(req("POST", sec_fetch_site="cross-site")).status_code == 403
    assert security.check(req("POST", origin="http://localhost:8765")) is None   # the agent's page


def test_uploads_must_declare_a_size_within_the_limit(client):
    r = client.post("/api/meetings/upload", content=b"x", headers={
        "content-type": "multipart/form-data; boundary=b", "content-length": str(3 * 1024**3)})
    assert r.status_code == 413


def test_local_notes_with_a_text_prompt_version_are_stored(client):
    """Regression: insight_runs.prompt_version is an int column; local runs label "2+local3"."""
    mid = client.post("/api/meetings/upload", files={"file": ("m.wav", _wav_bytes(), "audio/wav")},
                      data={"title": "Sync"}).json()["id"]
    local = {**INSIGHTS, "meta": {**INSIGHTS["meta"], "extractor": "qwen2.5:7b",
                                  "prompt_version": "2+local3", "cost_usd": 0.0}}
    db.store_results(mid, UTTS, local, 2.0)
    run = client.get(f"/api/meetings/{mid}/insights").json()
    assert run["meta"]["prompt_version"] == "2+local3"
    assert db.add_insight_run(mid, local)
