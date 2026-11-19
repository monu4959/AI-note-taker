"""Run every release gate and print one table. All gates are free: no paid LLM calls.

  G0  $0 guard     Claude blocked (NOTES_LLM=local) and the spend ledger unchanged
  G1  static       ruff (pyflakes + syntax), TypeScript check, web build
  G2  unit         pytest, no database / GPU / network
  G3  integration  pytest against the notetaker_test database (API, recorder, agent)
  G4  pipeline     real Whisper + pyannote + local LLM on a 2-minute clip: notes exist, cite
                   real lines, the run used no paid call
  G5  UI           Playwright end-to-end tests (tests/e2e)
  G6  quality      notes vs AMI human summaries on the dev meetings (free local grader);
                   needs a finished eval run: scripts/step1_eval_insights.py --tag <tag>
  G7  speed        full pipeline from scratch on a real meeting <= 25% of its length
  G8  package      the teammate recorder bundle builds and starts from a fresh unzip

Usage: .venv/Scripts/python scripts/gates.py [--only G0,G2] [--skip G7] [--g6-tag dev-qwen]
Results: artifacts/gates/last_run.json. Exit code 1 if any gate fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = sys.executable
OUT = ROOT / "artifacts" / "gates"
LEDGER = ROOT / "artifacts" / "llm_spend.jsonl"
BASELINE = OUT / "ledger_baseline.json"

UNIT = ["tests/test_assign.py", "tests/test_insights_checks.py", "tests/test_numbers.py",
        "tests/test_stages.py", "tests/test_speaker_id.py", "tests/test_recording.py",
        "tests/test_llm_local.py", "tests/test_judge_calibration.py"]
INTEGRATION = ["tests/test_api.py", "tests/test_recorder.py", "tests/test_agent.py"]

# G6 thresholds, fixed before looking at the test split (see reports/step4.md)
G6 = {"decisions_lenient": 0.80, "actions_lenient": 0.70, "wrong_number_rate_max": 0.05,
      "citation_agreement": 0.55}
G7_MAX_RATIO = 0.25          # pipeline time / audio length (15 min per hour)
RECORD_BASELINE = False


def sh(cmd: list[str] | str, cwd: Path = ROOT, timeout: int = 3600, env: dict | None = None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                       shell=isinstance(cmd, str), env={**os.environ, **(env or {})},
                       encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout + r.stderr)


def tail(text: str, n: int = 1) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:300]


def ledger_digest() -> str:
    return hashlib.sha256(LEDGER.read_bytes()).hexdigest() if LEDGER.exists() else "absent"


# --- gates ---------------------------------------------------------------------------------

def g0() -> tuple[bool, str]:
    from pipeline import llm

    if llm.PROVIDER != "local":
        return False, f"NOTES_LLM={llm.PROVIDER}: the $0 gate needs NOTES_LLM=local"
    try:
        llm.client()
        return False, "Claude client could be created"
    except llm.ClaudeDisabled:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    digest = ledger_digest()
    if not BASELINE.exists():
        if not RECORD_BASELINE:
            return False, "no ledger baseline: run once with --record-baseline (deliberately)"
        BASELINE.write_text(json.dumps({"sha256": digest, "recorded_at": datetime.now(UTC).isoformat()}))
        return True, f"Claude blocked; ledger baseline recorded ({digest[:12]})"
    base = json.loads(BASELINE.read_text())["sha256"]
    if base != digest:
        return False, "SPEND LEDGER CHANGED since the baseline: a paid call was made"
    # the running server must be in local mode too (it reads .env only when it starts)
    try:
        import httpx

        live = httpx.get("http://127.0.0.1:8000/api/config", timeout=5).json()["notes_llm"]
    except Exception:
        live = None
    if live not in (None, "local"):
        return False, f"the running API is in '{live}' mode: restart it after changing .env"
    return True, ("Claude blocked; spend ledger unchanged; running server "
                  + (f"in {live} mode" if live else "not running"))


def g1() -> tuple[bool, str]:
    rc1, o1 = sh([PY, "-m", "ruff", "check", "--select", "F,E9", "pipeline", "services",
                  "evaluation", "scripts", "tests"])
    rc2, o2 = sh("npx tsc --noEmit -p .", cwd=ROOT / "apps" / "web")
    rc3, o3 = sh("npm run build", cwd=ROOT / "apps" / "web")
    parts = [f"ruff {'ok' if rc1 == 0 else tail(o1, 3)}", f"tsc {'ok' if rc2 == 0 else tail(o2, 3)}",
             f"build {'ok' if rc3 == 0 else tail(o3, 3)}"]
    return rc1 == rc2 == rc3 == 0, ", ".join(parts)


def pytest_gate(paths: list[str], extra: list[str] | None = None) -> tuple[bool, str]:
    rc, out = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *(extra or []), *paths],
                 timeout=1800)
    summary = next((ln for ln in reversed(out.splitlines()) if " passed" in ln or " failed" in ln
                    or " error" in ln), tail(out))
    skipped = " skipped" in summary
    return rc == 0 and not skipped, summary.strip("= ") + (" (skips count as not green)" if skipped else "")


def g2():
    return pytest_gate(UNIT)


def g3():
    return pytest_gate(INTEGRATION)


def g5():
    return pytest_gate(["tests/e2e"])


def g4() -> tuple[bool, str]:
    """Fresh 2-minute clip through every stage, local LLM included."""
    import soundfile as sf

    from pipeline import ami, llm
    from pipeline.run import MEETINGS, PipelineConfig, insights_stage, names_stage, run

    mid = "gate-g4"
    shutil.rmtree(MEETINGS / mid, ignore_errors=True)
    wav, sr = sf.read(ami.audio_path("ES2004a"), dtype="float32", frames=16000 * 120)
    clip = OUT / "g4_clip.wav"
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(clip, wav, sr)
    before = ledger_digest()
    t0 = time.perf_counter()
    run(mid, clip, PipelineConfig(title="Gate G4 clip", name_cues=True))
    elapsed = time.perf_counter() - t0
    ins = json.loads((MEETINGS / mid / insights_stage(None) / "insights.json").read_text(encoding="utf-8"))
    utts = {u["id"] for u in json.loads((MEETINGS / mid / "assign" / "utterances.json").read_text(encoding="utf-8"))}
    names_ok = (MEETINGS / mid / names_stage() / "names.json").exists()
    bad = [it["id"] for it in ins["items"]
           if it["verification"]["status"] != "rejected"
           and not set(it["evidence"]["utterance_ids"]) <= utts]
    kept = [it for it in ins["items"] if it["verification"]["status"] != "rejected"]
    skipped = ins["meta"].get("skipped_windows", [])
    # store the real output the way the worker does, in the test database: catches schema
    # mismatches (e.g. a text prompt_version in an int column) that unit fixtures don't
    code = ("import json,sys; from services import db; db.apply_schema(); "
            "u=json.load(open(sys.argv[1],encoding='utf-8')); i=json.load(open(sys.argv[2],encoding='utf-8')); "
            "m=db.create_meeting('gate g4',[],None,'g4'); db.store_results(m,u,i,120.0); "
            "assert db.latest_insights(m)['payload']['meta']['extractor']==i['meta']['extractor']; "
            "db.delete_meeting(m); print('stored')")
    rc, out = sh([PY, "-c", code, str(MEETINGS / mid / "assign" / "utterances.json"),
                  str(MEETINGS / mid / insights_stage(None) / "insights.json")],
                 env={"DATABASE_URL": "postgresql://notetaker:notetaker@127.0.0.1:5434/notetaker_test"})
    stored = rc == 0 and "stored" in out
    ok = (bool(utts) and bool(kept) and not bad and bool(ins["summary"].strip()) and names_ok
          and not skipped and stored and ins["meta"]["extractor"] == llm.LOCAL_MODEL
          and ledger_digest() == before)
    return ok, (f"{len(utts)} utterances, {len(kept)} notes kept, summary {len(ins['summary'])} chars, "
                f"bad citations {len(bad)}, skipped windows {len(skipped)}, "
                f"stored in DB {'ok' if stored else 'FAILED: ' + tail(out, 2)}, "
                f"names stage {'ok' if names_ok else 'missing'}, "
                f"{elapsed:.0f}s, extractor {ins['meta']['extractor']}")


def g6(tag: str) -> tuple[bool, str]:
    from pipeline import llm

    if llm.is_local(llm.VERIFIER):
        # Measured (reports/step4.md): the local qwen2.5:7b grader agrees with the Sonnet
        # grader no better than chance (lenient kappa 0.08-0.13 on the dev meetings), so a
        # quality number from it would be meaningless. Grade with Sonnet instead (notes stay
        # local and free; only the grading is paid, a few cents per meeting).
        return False, ("BLOCKED: no trustworthy free grader (local judge kappa ~0.1). With "
                       "NOTES_LLM=claude: scripts/step1_eval_insights.py --tag dev-qwen "
                       "--extractor qwen2.5:7b IS1008d ES2011b TS3004c")
    path = ROOT / "artifacts" / "eval" / "runs" / f"{tag}.json"
    if not path.exists():
        return False, f"no eval run '{tag}' (run scripts/step1_eval_insights.py --tag {tag} ...)"
    rec = json.loads(path.read_text())
    p = rec["pooled"]

    def get(*keys):
        cur = p
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    vals = {"decisions_lenient": get("recall_decisions", "lenient"),
            "actions_lenient": get("recall_actions", "lenient"),
            "wrong_number_rate_max": get("wrong_number_rate", "value"),
            "citation_agreement": get("citation_agreement", "value")}
    fails = []
    for k, thr in G6.items():
        v = vals[k]
        if v is None:
            fails.append(f"{k} missing")
        elif (v > thr) if k.endswith("_max") else (v < thr):
            fails.append(f"{k} {v:.2f} vs {thr:.2f}")
    shown = ", ".join(f"{k} {v:.2f}" for k, v in vals.items() if v is not None)
    return not fails, f"[{tag}, judge {rec.get('verifier_judge')}] {shown}" + (
        f" | FAIL: {'; '.join(fails)}" if fails else "")


def g7() -> tuple[bool, str]:
    """A real meeting from scratch (new meeting id, so nothing is cached)."""
    from pipeline import ami
    from pipeline.run import MEETINGS, PipelineConfig, run

    m = "IS1009c"
    mid = "gate-g7"
    shutil.rmtree(MEETINGS / mid, ignore_errors=True)
    t0 = time.perf_counter()
    run(mid, ami.audio_path(m), PipelineConfig(title=f"Gate G7 {m}", name_cues=True))
    elapsed = time.perf_counter() - t0
    dur = json.loads((MEETINGS / mid / "prepare" / "manifest.json").read_text())["info"]["duration_s"]
    ratio = elapsed / dur
    return ratio <= G7_MAX_RATIO, (f"{m}: {dur / 60:.0f} min of audio in {elapsed / 60:.1f} min "
                                   f"(ratio {ratio:.3f}, limit {G7_MAX_RATIO})")


def g8() -> tuple[bool, str]:
    build = Path(tempfile.mkdtemp(prefix="gate-g8-build-"))   # never touch dist/ (real bundle)
    rc, out = sh([PY, "scripts/make_recorder_bundle.py", "--server", "http://127.0.0.1:8000",
                  "--out", str(build)], timeout=900)
    zpath = build / "notetaker-recorder.zip"
    if rc != 0 or not zpath.exists():
        return False, "bundle build failed: " + tail(out, 3)
    tmp = Path(tempfile.mkdtemp(prefix="gate-g8-"))
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        root = tmp / "notetaker-recorder"
        import socket

        with socket.socket() as so:          # a free port: never talk to a leftover process
            so.bind(("127.0.0.1", 0))
            port = str(so.getsockname()[1])
        env = {"NOTETAKER_SERVER": "http://127.0.0.1:8000", "NOTETAKER_WEB": str(root / "web"),
               "NOTETAKER_AGENT_PORT": port, "VIRTUAL_ENV": ""}
        proc = subprocess.Popen(["uv", "run", "--quiet", "python", "-m", "services.agent"],
                                cwd=root, env={**os.environ, **env},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            import httpx

            for _ in range(90):
                try:
                    page = httpx.get(f"http://127.0.0.1:{port}/", timeout=2)
                    info = httpx.get(f"http://127.0.0.1:{port}/api/agent", timeout=2).json()
                    rec = httpx.get(f"http://127.0.0.1:{port}/api/recorder", timeout=10).json()
                    ok = page.status_code == 200 and info.get("agent") and "state" in rec
                    return bool(ok), (f"{zpath.stat().st_size / 1e6:.1f} MB zip; fresh unzip starts; "
                                      f"page {page.status_code}, recorder {rec.get('state')}")
                except (httpx.HTTPError, ValueError):
                    time.sleep(1)
            return False, "the unzipped recorder did not answer within 90 s"
        finally:   # `uv run` starts python as a child: stop the whole tree
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
            proc.wait(timeout=20)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(build, ignore_errors=True)


GATES = {"G0": ("$0 guard", g0), "G1": ("static", g1), "G2": ("unit", g2),
         "G3": ("integration", g3), "G4": ("pipeline", g4), "G5": ("UI e2e", g5),
         "G6": ("quality", None), "G7": ("speed", g7), "G8": ("package", g8)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--g6-tag", default="dev-qwen")
    ap.add_argument("--record-baseline", action="store_true",
                    help="record the spend ledger's current state as the $0 baseline")
    a = ap.parse_args()
    global RECORD_BASELINE
    RECORD_BASELINE = a.record_baseline
    only = {g.strip().upper() for g in a.only.split(",") if g.strip()}
    skip = {g.strip().upper() for g in a.skip.split(",") if g.strip()}
    results = []
    for gid, (name, fn) in GATES.items():
        if (only and gid not in only) or gid in skip:
            continue
        t0 = time.perf_counter()
        try:
            ok, detail = g6(a.g6_tag) if gid == "G6" else fn()
        except Exception as e:  # a crashing gate is a failing gate
            ok, detail = False, f"{type(e).__name__}: {e}"[:300]
        results.append({"gate": gid, "name": name, "ok": ok, "detail": detail,
                        "seconds": round(time.perf_counter() - t0)})
        print(f"{gid} {name:12s} {'GREEN' if ok else 'RED  '} {results[-1]['seconds']:>5}s  {detail}",
              flush=True)
    if not only or "G0" in only:   # re-check the ledger last: nothing above may have spent money
        ok, detail = g0()
        print(f"G0 (final)    {'GREEN' if ok else 'RED  '}         {detail}")
        results.append({"gate": "G0-final", "name": "$0 guard (after all gates)", "ok": ok,
                        "detail": detail, "seconds": 0})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "last_run.json").write_text(json.dumps(
        {"at": datetime.now(UTC).isoformat(), "results": results}, indent=1))
    green = all(r["ok"] for r in results)
    print("ALL GREEN" if green else "NOT GREEN: " + ", ".join(r["gate"] for r in results if not r["ok"]))
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
