"""Postgres access shared by the API and the worker."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

import pipeline  # noqa: F401  (.env)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "infra" / "schema.sql"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://notetaker:notetaker@127.0.0.1:5434/notetaker")


@contextmanager
def connect():
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        yield conn


def apply_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA.read_text())


# --- meetings & jobs -------------------------------------------------------------------

def create_meeting(title: str, attendees: list[str], num_speakers: int | None,
                   source_name: str) -> str:
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO meetings (title, attendees, num_speakers, source_name) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (title, Jsonb(attendees), num_speakers, source_name)).fetchone()
        return str(row["id"])


def create_recording(title: str, attendees: list[str], owner_name: str | None,
                     source_name: str) -> str:
    """A meeting recorded on this computer; its audio arrives (and its job is queued) once
    the recorder has mixed the tracks."""
    with connect() as conn:
        row = conn.execute(
            "INSERT INTO meetings (title, attendees, source_name, mode, owner_name) "
            "VALUES (%s, %s, %s, 'live', %s) RETURNING id",
            (title, Jsonb(attendees), source_name, owner_name)).fetchone()
        return str(row["id"])


def set_upload(meeting_id: str, path: Path) -> None:
    with connect() as conn:
        conn.execute("UPDATE meetings SET audio_uris = audio_uris || %s WHERE id = %s",
                     (Jsonb({"upload": str(path)}), meeting_id))
        conn.execute("INSERT INTO jobs (meeting_id) VALUES (%s)", (meeting_id,))


def claim_job() -> dict | None:
    """Atomically take the oldest queued job (safe with several workers)."""
    with connect() as conn:
        return conn.execute(
            """UPDATE jobs SET status = 'running', started_at = now(), attempts = attempts + 1
               WHERE id = (SELECT id FROM jobs WHERE status = 'queued'
                           ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING *""").fetchone()


def requeue_stale_jobs() -> int:
    """A single worker restarting means any 'running' job was interrupted."""
    with connect() as conn:
        return conn.execute(
            "UPDATE jobs SET status = 'queued', stage = NULL WHERE status = 'running'").rowcount


def update_job(job_id: int, **fields) -> None:
    cols = ", ".join(f"{k} = %s" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = %s", (*fields.values(), job_id))


def update_meeting(meeting_id: str, **fields) -> None:
    cols = ", ".join(f"{k} = %s" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE meetings SET {cols} WHERE id = %s", (*fields.values(), meeting_id))


def get_meeting(meeting_id: str) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM meetings WHERE id = %s", (meeting_id,)).fetchone()


def latest_job(meeting_id: str) -> dict | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE meeting_id = %s ORDER BY id DESC LIMIT 1",
                            (meeting_id,)).fetchone()


def list_meetings() -> list[dict]:
    with connect() as conn:
        return conn.execute(
            """SELECT m.id, m.title, m.status, m.duration_s, m.created_at, m.source_name,
                      m.mode, j.stage, j.progress
               FROM meetings m
               LEFT JOIN LATERAL (SELECT stage, progress FROM jobs WHERE meeting_id = m.id
                                  ORDER BY id DESC LIMIT 1) j ON true
               ORDER BY m.created_at DESC""").fetchall()


def delete_meeting(meeting_id: str) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM meetings WHERE id = %s", (meeting_id,)).rowcount > 0


# --- results ---------------------------------------------------------------------------

def _seq(utt_id: str) -> int:
    return int(utt_id.lstrip("u"))


def store_results(meeting_id: str, utterances: list[dict], insights: dict | None,
                  duration_s: float) -> None:
    """Replace the meeting's final transcript, speakers and insights in one transaction."""
    with connect() as conn, conn.transaction():
        conn.execute("DELETE FROM utterances WHERE meeting_id = %s AND pass = 'final'",
                     (meeting_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO utterances (meeting_id, seq, speaker_label, start_ms, end_ms, text,"
                " words, pass, asr_confidence) VALUES (%s, %s, %s, %s, %s, %s, %s, 'final', %s)",
                [(meeting_id, _seq(u["id"]), u["speaker"], round(u["start"] * 1000),
                  round(u["end"] * 1000), u["text"],
                  Jsonb([{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2),
                          "p": round(w.get("prob", 1.0), 3)} for w in u.get("words", [])]),
                  _mean_prob(u)) for u in utterances])
        labels = sorted({u["speaker"] for u in utterances})
        for label in labels:  # keep any name the user already gave this cluster
            conn.execute(
                "INSERT INTO speakers (meeting_id, cluster_label) VALUES (%s, %s) "
                "ON CONFLICT (meeting_id, cluster_label) DO NOTHING", (meeting_id, label))
        conn.execute("DELETE FROM speakers WHERE meeting_id = %s AND NOT (cluster_label = ANY(%s))",
                     (meeting_id, labels))

        if insights is not None:
            _insert_insights(conn, meeting_id, insights)
        conn.execute("UPDATE meetings SET status = 'ready', duration_s = %s, ended_at = now(),"
                     " error = NULL WHERE id = %s", (duration_s, meeting_id))


def _insert_insights(conn, meeting_id: str, insights: dict) -> int:
    meta = insights["meta"]
    # the column holds the base (Claude) prompt version; local runs label theirs "2+local3"
    # (full label kept in the payload's meta)
    pv = meta["prompt_version"]
    base_pv = pv if isinstance(pv, int) else int(str(pv).split("+", 1)[0])
    run_id = conn.execute(
        "INSERT INTO insight_runs (meeting_id, model, prompt_version, schema_version,"
        " payload) VALUES (%s, %s, %s, %s, %s) RETURNING id",
        (meeting_id, meta["extractor"], base_pv, meta["schema_version"],
         Jsonb(insights))).fetchone()["id"]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO insight_items (run_id, item_key, type, content, evidence_seqs,"
            " status) VALUES (%s, %s, %s, %s, %s, %s)",
            [(run_id, it["id"], it["type"], Jsonb(it),
              [_seq(i) for i in it["evidence"]["utterance_ids"] if i.startswith("u")],
              it["verification"]["status"]) for it in insights["items"]])
    return run_id


def _mean_prob(u: dict) -> float | None:
    probs = [w.get("prob") for w in u.get("words", []) if w.get("prob") is not None]
    return round(sum(probs) / len(probs), 4) if probs else None


def transcript(meeting_id: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT seq, speaker_label, start_ms, end_ms, text, asr_confidence FROM utterances"
            " WHERE meeting_id = %s AND pass = 'final' ORDER BY seq", (meeting_id,)).fetchall()


def speakers(meeting_id: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "SELECT cluster_label, display_name, method, confirmed, confidence, evidence,"
            " person_id FROM speakers WHERE meeting_id = %s ORDER BY cluster_label",
            (meeting_id,)).fetchall()


def rename_speaker(meeting_id: str, label: str, display_name: str | None) -> bool:
    with connect() as conn:
        return conn.execute(
            "UPDATE speakers SET display_name = %s, method = CASE WHEN %s::text IS NULL THEN NULL"
            " ELSE 'manual' END WHERE meeting_id = %s AND cluster_label = %s",
            (display_name, display_name, meeting_id, label)).rowcount > 0


def latest_insights(meeting_id: str, run_id: int | None = None) -> dict | None:
    """One insight run: `run_id`, else the newest."""
    with connect() as conn:
        if run_id is not None:
            return conn.execute(
                "SELECT id, model, prompt_version, schema_version, payload, created_at"
                " FROM insight_runs WHERE meeting_id = %s AND id = %s", (meeting_id, run_id)).fetchone()
        return conn.execute(
            "SELECT id, model, prompt_version, schema_version, payload, created_at"
            " FROM insight_runs WHERE meeting_id = %s ORDER BY id DESC LIMIT 1",
            (meeting_id,)).fetchone()


def insight_runs(meeting_id: str) -> list[dict]:
    """The newest run per model, newest first, with what the UI needs to compare them."""
    with connect() as conn:
        return conn.execute(
            "SELECT DISTINCT ON (model) id, model, created_at,"
            " (payload->'meta'->>'cost_usd')::float AS cost_usd,"
            " (payload->'meta'->>'items')::int AS items, payload->'meta'->'status' AS status"
            " FROM insight_runs WHERE meeting_id = %s ORDER BY model, id DESC",
            (meeting_id,)).fetchall()


def enqueue_notes(meeting_id: str, extractor: str) -> dict:
    """Queue a notes-only job with another model (transcript and speakers are reused)."""
    with connect() as conn:
        return conn.execute("INSERT INTO jobs (meeting_id, params) VALUES (%s, %s) RETURNING *",
                            (meeting_id, Jsonb({"extractor": extractor}))).fetchone()


def add_insight_run(meeting_id: str, insights: dict) -> int:
    with connect() as conn, conn.transaction():
        return _insert_insights(conn, meeting_id, insights)
