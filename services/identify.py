"""Name a meeting's speakers from all signals, and enroll / forget voices.

Order of precedence for each speaker row:
  1. confirmed by the user           never changed here
  2. your microphone                 recordings made on this computer: the one cluster whose
                                     speech is mostly mic-only is you (pipeline/recording.py)
  3. voiceprint match >= threshold   (pipeline/speaker_id.py, calibrated on AMI)
  4. transcript cue                  (pipeline/name_cues.py), only if no other speaker in the
                                     meeting already carries that name
  otherwise anonymous ("Speaker N" in the UI).

Voiceprints are biometric data: one is stored only when the user confirms a speaker with
"remember this voice", and forgetting a person deletes every stored embedding.
"""

from __future__ import annotations

import json
import os

import numpy as np
from psycopg.types.json import Jsonb

from pipeline import recording, speaker_id
from pipeline.run import MEETINGS
from services import db


class EnrollError(ValueError):
    pass


def _clusters(meeting_id: str) -> list[speaker_id.Cluster]:
    return speaker_id.clusters_from_diarization(MEETINGS / meeting_id / "diarize")


def _cues(meeting_id: str) -> dict:
    """Accepted transcript cues: the active model's, else any earlier run's (e.g. Claude's)."""
    from pipeline.run import names_stage

    base = MEETINGS / meeting_id
    paths = [base / names_stage() / "names.json", base / "names" / "names.json",
             *sorted(base.glob("names-*/names.json"))]
    path = next((p for p in paths if p.exists()), None)
    return json.loads(path.read_text(encoding="utf-8"))["accepted"] if path else {}


def owner(meeting: dict) -> tuple[str, dict] | None:
    """(cluster label, evidence) of the person who recorded the meeting, if it is clear."""
    levels = (meeting.get("audio_uris") or {}).get("levels")
    dia = MEETINGS / str(meeting["id"]) / "diarize" / "diarization.json"
    if not levels or not dia.exists() or not os.path.exists(levels):
        return None
    lv = np.load(levels)
    turns = [(t["start"], t["end"], t["speaker"])
             for t in json.loads(dia.read_text(encoding="utf-8"))["exclusive"]]
    label, evidence = recording.owner_from_levels(turns, lv)
    return (label, evidence) if label else None


def gallery(conn) -> tuple[dict[str, list[np.ndarray]], dict[str, dict]]:
    """Enrolled voices for the current embedding model: ({person_id: [vectors]}, {id: person})."""
    rows = conn.execute(
        "SELECT p.id, p.display_name, v.embedding::text AS e FROM voiceprints v"
        " JOIN persons p ON p.id = v.person_id WHERE v.model_name = %s",
        (speaker_id.EMBEDDING_MODEL,)).fetchall()
    g: dict[str, list[np.ndarray]] = {}
    people = {}
    for r in rows:
        g.setdefault(str(r["id"]), []).append(np.array(json.loads(r["e"])))
        people[str(r["id"])] = {"id": r["id"], "display_name": r["display_name"]}
    return g, people


def identify_meeting(meeting_id: str) -> dict:
    """Re-derive automatic names for one meeting. Cheap (no GPU, no LLM)."""
    clusters = _clusters(meeting_id)
    cues = _cues(meeting_id)
    with db.connect() as conn, conn.transaction():
        m = conn.execute("SELECT id, attendees, audio_uris, owner_name FROM meetings"
                         " WHERE id = %s", (meeting_id,)).fetchone()
        rows = {r["cluster_label"]: r for r in conn.execute(
            "SELECT cluster_label, display_name, confirmed FROM speakers WHERE meeting_id = %s",
            (meeting_id,)).fetchall()}
        g, people = gallery(conn)
        attendees = {a.lower() for a in (m["attendees"] if m else [])}
        candidates = ({pid for pid, p in people.items()
                       if p["display_name"].lower() in attendees} or None) if attendees else None
        # people already confirmed in this meeting are not candidates for other clusters
        taken = {r["display_name"].lower() for r in rows.values()
                 if r["confirmed"] and r["display_name"]}
        names: dict[str, tuple] = {}
        me = owner(m) if m else None
        if me and me[0] in rows and not rows[me[0]]["confirmed"]:
            my_name = m["owner_name"] or "You"
            if my_name.lower() not in taken:
                names[me[0]] = (my_name, None, me[1]["share"], "mic", me[1])
                taken = taken | {my_name.lower()}
        g = {pid: v for pid, v in g.items() if people[pid]["display_name"].lower() not in taken}
        open_clusters = [c for c in clusters if c.label in rows
                         and not rows[c.label]["confirmed"] and c.label not in names]
        matches = {mt.label: mt for mt in speaker_id.identify(open_clusters, g,
                                                              candidates=candidates)}

        for label, mt in matches.items():
            p = people[mt.person]
            names[label] = (p["display_name"], p["id"], mt.score, "voiceprint",
                            {"score": mt.score, "runner_up": mt.runner_up,
                             "threshold": speaker_id.THRESHOLD})
        used = taken | {n[0].lower() for n in names.values()}
        for label, cue in cues.items():
            if label in names or label not in rows or rows[label]["confirmed"]:
                continue
            if cue["name"].lower() in used:
                continue
            names[label] = (cue["name"], None, None, "llm_cue",
                            {"kinds": cue["kinds"], "utterance_ids": cue["evidence"]})
            used.add(cue["name"].lower())

        for label, r in rows.items():
            if r["confirmed"]:
                continue
            n = names.get(label)
            conn.execute(
                "UPDATE speakers SET display_name = %s, person_id = %s, confidence = %s,"
                " method = %s, evidence = %s WHERE meeting_id = %s AND cluster_label = %s",
                (n[0], n[1], n[2], n[3], Jsonb(n[4]), meeting_id, label) if n else
                (None, None, None, None, None, meeting_id, label))
    return {"mic": sum(n[3] == "mic" for n in names.values()),
            "voiceprint": sum(n[3] == "voiceprint" for n in names.values()),
            "llm_cue": sum(n[3] == "llm_cue" for n in names.values())}


def identify_all() -> None:
    with db.connect() as conn:
        ids = [str(r["id"]) for r in conn.execute(
            "SELECT id FROM meetings WHERE status = 'ready'").fetchall()]
    for mid in ids:
        identify_meeting(mid)


def confirm(meeting_id: str, label: str, name: str | None, remember_voice: bool) -> dict:
    """The user names (or un-names) a speaker. With remember_voice, enroll the voice."""
    with db.connect() as conn, conn.transaction():
        row = conn.execute("SELECT 1 FROM speakers WHERE meeting_id = %s AND cluster_label = %s",
                           (meeting_id, label)).fetchone()
        if row is None:
            raise KeyError(label)
        if name is None:  # back to anonymous; let automatic signals try again
            conn.execute(
                "UPDATE speakers SET display_name = NULL, person_id = NULL, method = NULL,"
                " confidence = NULL, evidence = NULL, confirmed = false"
                " WHERE meeting_id = %s AND cluster_label = %s", (meeting_id, label))
        else:
            person_id = None
            if remember_voice:
                cluster = next((c for c in _clusters(meeting_id) if c.label == label), None)
                if cluster is None or cluster.speech_s < speaker_id.MIN_SPEECH_S:
                    raise EnrollError(
                        f"not enough clear speech from this speaker to remember the voice "
                        f"(needs {speaker_id.MIN_SPEECH_S:.0f} s)")
                person_id = conn.execute(
                    "INSERT INTO persons (display_name) VALUES (%s) ON CONFLICT"
                    " ((lower(display_name))) DO UPDATE SET display_name = persons.display_name"
                    " RETURNING id", (name,)).fetchone()["id"]
                conn.execute("DELETE FROM voiceprints WHERE source_meeting_id = %s"
                             " AND source_label = %s", (meeting_id, label))
                conn.execute(
                    "INSERT INTO voiceprints (person_id, embedding, model_name, source_meeting_id,"
                    " source_label, speech_s) VALUES (%s, %s::vector, %s, %s, %s, %s)",
                    (person_id, json.dumps(cluster.embedding.round(6).tolist()),
                     speaker_id.EMBEDDING_MODEL, meeting_id, label, cluster.speech_s))
            conn.execute(
                "UPDATE speakers SET display_name = %s, person_id = %s, method = 'manual',"
                " confidence = NULL, evidence = NULL, confirmed = true"
                " WHERE meeting_id = %s AND cluster_label = %s",
                (name, person_id, meeting_id, label))
        # a name the user just gave can't stay on another, automatically named cluster
        if name:
            conn.execute(
                "UPDATE speakers SET display_name = NULL, person_id = NULL, method = NULL,"
                " confidence = NULL, evidence = NULL WHERE meeting_id = %s AND cluster_label <> %s"
                " AND NOT confirmed AND lower(display_name) = lower(%s)",
                (meeting_id, label, name))
    if name and remember_voice:
        identify_all()   # the new voice may name speakers in other meetings
    else:
        identify_meeting(meeting_id)
    return {"cluster_label": label, "display_name": name, "remembered": bool(name and remember_voice)}


def people() -> list[dict]:
    with db.connect() as conn:
        return conn.execute(
            "SELECT p.id, p.display_name, count(DISTINCT v.id) AS voiceprints,"
            " count(DISTINCT s.meeting_id) AS meetings, min(v.consented_at) AS remembered_since"
            " FROM persons p LEFT JOIN voiceprints v ON v.person_id = p.id"
            " LEFT JOIN speakers s ON s.person_id = p.id"
            " GROUP BY p.id ORDER BY lower(p.display_name)").fetchall()


def forget(person_id: int) -> bool:
    """Delete a person and every stored voiceprint; their automatic names disappear."""
    with db.connect() as conn, conn.transaction():
        conn.execute("UPDATE speakers SET display_name = NULL, method = NULL, confidence = NULL,"
                     " evidence = NULL WHERE person_id = %s AND NOT confirmed", (person_id,))
        gone = conn.execute("DELETE FROM persons WHERE id = %s", (person_id,)).rowcount > 0
    if gone:
        identify_all()
    return gone
