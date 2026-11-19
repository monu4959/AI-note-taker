"""Cached, idempotent pipeline stages.

Each stage writes into artifacts/meetings/<meeting_id>/<stage>/ plus a manifest.json holding a
cache key = hash(stage name, stage version, stage params, upstream keys). Re-running a meeting
skips every stage whose key is unchanged, so e.g. a new insights prompt never re-transcribes,
and swapping the ASR model re-runs transcription and everything downstream of it only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path


def _key(name: str, version: int, params: dict, deps: dict[str, str]) -> str:
    blob = json.dumps({"name": name, "version": version, "params": params, "deps": deps},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class MeetingStore:
    def __init__(self, root: Path, meeting_id: str):
        self.meeting_id = meeting_id
        self.root = root / meeting_id

    def dir(self, stage: str) -> Path:
        return self.root / stage

    def manifest(self, stage: str) -> dict | None:
        path = self.dir(stage) / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else None

    def read_json(self, stage: str, name: str):
        return json.loads((self.dir(stage) / name).read_text(encoding="utf-8"))


def _output_fingerprint(out: Path) -> str:
    """Hash of the stage's output files, so downstream stages re-run iff the content changed."""
    h = hashlib.sha256()
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            h.update(p.relative_to(out).as_posix().encode())
            h.update(file_sha256(p).encode())
    return h.hexdigest()[:16]


def run_stage(store: MeetingStore, name: str, version: int, params: dict,
              deps: dict[str, str], fn: Callable[[Path], dict | None],
              on_event: Callable[[dict], None] | None = None) -> str:
    """Run `fn(out_dir)` unless a finished run with the same key exists.

    The cache key covers (name, version, params, upstream fingerprints). The return value is
    this stage's *output* fingerprint, which downstream stages put in their `deps`: a re-run
    that reproduces identical output leaves downstream cached; different output re-runs it.
    """
    key = _key(name, version, params, deps)
    emit = on_event or (lambda e: None)
    m = store.manifest(name)
    if m and m.get("key") == key and m.get("status") == "done":
        emit({"stage": name, "status": "cached", "key": key})
        return m["fingerprint"]

    out = store.dir(name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    emit({"stage": name, "status": "running", "key": key})
    t0 = time.perf_counter()
    info = fn(out) or {}
    fingerprint = _output_fingerprint(out)
    manifest = {
        "stage": name, "version": version, "key": key, "fingerprint": fingerprint,
        "params": params, "deps": deps,
        "status": "done", "elapsed_s": round(time.perf_counter() - t0, 2),
        "finished_at": datetime.now(UTC).isoformat(), "info": info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    emit({"stage": name, "status": "done", "key": key, "elapsed_s": manifest["elapsed_s"]})
    return fingerprint


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]
