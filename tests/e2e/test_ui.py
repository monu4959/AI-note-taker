"""G5: the web app end to end in a real browser (system Microsoft Edge via Playwright).

Self-contained: starts its own API server on a free port against the test database
(notetaker_test), serves the built app from apps/web/dist (rebuilt here if older than the
source), seeds meetings straight through services/db.py, and deletes them afterwards. No
worker, GPU or LLM is involved. Skipped if Postgres or Edge is unavailable.

    .venv/Scripts/python -m pytest tests/e2e -q        # about 20 s
    E2E_SCREENSHOTS=<dir> ...                           # also save 1440 px and 390 px shots
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

TEST_DB = "postgresql://notetaker:notetaker@127.0.0.1:5434/notetaker_test"
os.environ["DATABASE_URL"] = TEST_DB

from psycopg.types.json import Jsonb  # noqa: E402

from services import db  # noqa: E402

pytestmark = pytest.mark.e2e

# .env is loaded with override=True: never seed or delete rows in a database we didn't pick.
if db.DATABASE_URL != TEST_DB:
    pytest.skip("DATABASE_URL points away from the test database", allow_module_level=True)
try:
    db.apply_schema()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Postgres not available: {exc}", allow_module_level=True)
try:
    from playwright.sync_api import Browser, Page, Route, expect, sync_playwright
except ImportError:  # pragma: no cover
    pytest.skip("playwright is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "web"
DURATION = 30.0
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))

UTTS = [
    {"id": "u0000", "speaker": "SPEAKER_00", "start": 0.0, "end": 4.0,
     "text": "Morning everyone, let's plan the quarter."},
    {"id": "u0001", "speaker": "SPEAKER_01", "start": 4.5, "end": 11.0,
     "text": "The marketing budget is forty thousand dollars."},
    {"id": "u0002", "speaker": "SPEAKER_00", "start": 12.0, "end": 18.0,
     "text": "Okay, we'll ship the beta on March 3."},
    {"id": "u0003", "speaker": "SPEAKER_01", "start": 18.5, "end": 24.0,
     "text": "I'll send the launch draft by Friday."},
    {"id": "u0004", "speaker": "SPEAKER_00", "start": 24.5, "end": 29.5,
     "text": "Who owns the pricing page, though?"},
]

OK = {"status": "verified", "support": "supported", "reason": None, "checks": []}
CHECK = {"status": "needs_review", "support": "partial", "reason": "The due day is not explicit.",
         "checks": []}


def _notes(model: str, cost: float, summary: str) -> dict:
    items = [
        {"id": "d1", "type": "decision", "topic_id": "t1", "text": "Beta ships on March 3",
         "evidence": {"utterance_ids": ["u0002"]}, "verification": OK},
        {"id": "a1", "type": "action", "topic_id": "t1", "task": "Send the launch draft",
         "owner": "SPEAKER_01", "due": "Friday",
         "evidence": {"utterance_ids": ["u0003"]}, "verification": CHECK},
        {"id": "m1", "type": "metric", "topic_id": "t1", "name": "Marketing budget",
         "value": "40,000", "unit": "USD", "context": None,
         "evidence": {"utterance_ids": ["u0001"]}, "verification": OK},
        {"id": "q1", "type": "open_question", "topic_id": "t1",
         "text": "Who owns the pricing page?",
         "evidence": {"utterance_ids": ["u0004"]}, "verification": OK},
        {"id": "k1", "type": "key_point", "topic_id": "t1", "text": "The quarter plan is set",
         "evidence": {"utterance_ids": ["u0000"]}, "verification": OK},
        {"id": "qa1", "type": "qa", "topic_id": "t1", "question": "When does the beta ship?",
         "asked_by": "SPEAKER_01", "answer": "On March 3.", "answered_by": "SPEAKER_00",
         "status": "answered", "evidence": {"utterance_ids": ["u0001", "u0002"]},
         "verification": OK},
    ]
    return {
        "meta": {"extractor": model, "verifier": model, "prompt_version": 2, "schema_version": 1,
                 "status": {"verified": 4, "needs_review": 1, "rejected": 0},
                 "cost_usd": cost, "items": len(items)},
        "summary": summary,
        "topics": [{"id": "t1", "title": "Quarter plan", "start_id": "u0000", "end_id": "u0004"}],
        "items": items,
    }


QWEN_SUMMARY = "The team set the quarter plan: the beta ships on March 3 (Qwen notes)."
SONNET_SUMMARY = "Quarter planning: beta on March 3 (Sonnet notes)."
OPUS_SUMMARY = "An older meeting whose notes were written by Opus."

LOCAL_CONFIG = {"notes_llm": "local", "extractor": "qwen2.5:7b", "verifier": "qwen2.5:7b",
                "note_models": {"local": "qwen2.5:7b"}}
CLAUDE_CONFIG = {"notes_llm": "claude", "extractor": "claude-opus-5-5",
                 "verifier": "claude-sonnet-5-5",
                 "note_models": {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}}


# --- fixtures ----------------------------------------------------------------------------

def _ensure_dist() -> None:
    """The API serves apps/web/dist; rebuild it if any web source is newer."""
    index = WEB / "dist" / "index.html"
    sources = [p for p in (WEB / "src").rglob("*") if p.is_file()]
    sources += [WEB / n for n in ("index.html", "package.json", "vite.config.ts") if (WEB / n).exists()]
    if index.exists() and index.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return
    npm = shutil.which("npm")
    if not npm or not (WEB / "node_modules").exists():
        pytest.skip("apps/web/dist is missing or older than the source: run `npm run build` in apps/web")
    r = subprocess.run([npm, "run", "build"], cwd=WEB, capture_output=True, text=True, timeout=240)
    if r.returncode:
        pytest.skip(f"`npm run build` failed in apps/web:\n{(r.stdout + r.stderr)[-2000:]}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str):
    with NO_PROXY.open(url, timeout=5) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge")
        except Exception as exc:
            pytest.skip(f"Microsoft Edge is not available to Playwright: {str(exc).splitlines()[0]}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def server(browser, tmp_path_factory):
    """uvicorn in a subprocess (its own DATABASE_URL, never the user's server on :8000)."""
    _ensure_dist()
    port = _free_port()
    log = tmp_path_factory.mktemp("server") / "api.log"
    env = {**os.environ, "DATABASE_URL": TEST_DB, "PYTHONUNBUFFERED": "1"}
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "services.api:app", "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 50
        while True:
            if proc.poll() is not None:
                pytest.fail(f"API server exited:\n{log.read_text(encoding='utf-8')[-3000:]}")
            try:
                if _get_json(base + "/api/health").get("ok"):
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"API server did not start:\n{log.read_text(encoding='utf-8')[-3000:]}")
            time.sleep(0.25)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _seed(ids: list[str], title: str, audio: Path, runs: list[dict] | None,
          job: dict | None = None) -> str:
    """A meeting as the worker leaves it: prepared audio, transcript, speakers, notes runs."""
    mid = db.create_meeting(title, ["Priya", "Raj"], 2, "e2e.wav")
    ids.append(mid)
    db.set_upload(mid, audio)
    with db.connect() as conn:
        conn.execute("UPDATE meetings SET audio_uris = audio_uris || %s WHERE id = %s",
                     (Jsonb({"prepared": str(audio)}), mid))
    job_id = db.latest_job(mid)["id"]
    if runs is None:  # still processing
        db.update_meeting(mid, status="processing")
        db.update_job(job_id, **(job or {}))
        return mid
    db.store_results(mid, UTTS, runs[0], duration_s=DURATION)
    for r in runs[1:]:
        db.add_insight_run(mid, r)
    db.update_job(job_id, status="done", stage="insights", progress=1.0)
    db.rename_speaker(mid, "SPEAKER_00", "Priya")
    return mid


@pytest.fixture(scope="module")
def meetings(server, tmp_path_factory):
    audio = tmp_path_factory.mktemp("audio") / "prepared.wav"
    t = np.arange(int(16000 * DURATION)) / 16000
    sf.write(audio, (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000,
             subtype="PCM_16")
    ids: list[str] = []
    try:
        seeded = {
            "main": _seed(ids, "E2E quarter planning", audio, [
                _notes("claude-sonnet-5-5", 0.21, SONNET_SUMMARY),
                _notes("qwen2.5:7b", 0.0, QWEN_SUMMARY)]),
            "opus": _seed(ids, "E2E older Opus meeting", audio, [
                _notes("claude-opus-5-5", 0.41, OPUS_SUMMARY)]),
            "busy": _seed(ids, "E2E still processing", audio, None,
                          {"status": "running", "stage": "insights", "progress": 0.8}),
        }
        # the server reads the same database we seeded (guards against a different DB URL)
        assert _get_json(f"{server}/api/meetings/{seeded['main']}")["meeting"]["status"] == "ready"
        yield seeded
    finally:
        for mid in ids:
            db.delete_meeting(mid)


class Tab:
    """A browser page plus the console errors it logged."""

    def __init__(self, browser: Browser, base: str, width: int, config: dict | int | None = None):
        height = 844 if width < 600 else 900
        self.ctx = browser.new_context(base_url=base, viewport={"width": width, "height": height})
        self.page: Page = self.ctx.new_page()
        self.errors: list[str] = []
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        if config is not None:   # pin the provider: a dict is served, an int is an error status
            def fulfil(route: Route):
                if isinstance(config, int):
                    route.fulfill(status=config, body="unavailable")
                else:
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(config))
            self.page.route("**/api/config", fulfil)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.ctx.close()


def _write_offers(page: Page) -> list[str]:
    return page.get_by_role("button", name=re.compile(r"write notes with", re.I)).all_inner_texts()


def _no_horizontal_scroll(page: Page, width: int) -> None:
    overflow = page.evaluate("""(width) => {
        const bad = [];
        if (document.documentElement.scrollWidth > width) bad.push(`document ${document.documentElement.scrollWidth}`);
        for (const el of document.querySelectorAll('*')) {
            const ox = getComputedStyle(el).overflowX;
            if ((ox === 'auto' || ox === 'scroll') && el.scrollWidth > el.clientWidth + 1)
                bad.push(`${el.tagName.toLowerCase()}.${el.className.toString().slice(0, 60)} ${el.scrollWidth}>${el.clientWidth}`);
        }
        return bad;
    }""", width)
    assert overflow == [], f"horizontal scroll at {width} px: {overflow}"


def _shot(page: Page, name: str) -> None:
    out = os.environ.get("E2E_SCREENSHOTS")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(out) / name))


# --- tests -------------------------------------------------------------------------------

def test_meeting_list_notes_transcript_player_and_speakers(browser, server, meetings):
    config = _get_json(server + "/api/config")   # whatever NOTES_LLM this machine runs
    with Tab(browser, server, 1440) as tab:
        page = tab.page
        page.goto("/#/")
        link = page.get_by_role("link", name=re.compile("E2E quarter planning"))
        expect(link).to_be_visible()
        link.click()
        expect(page).to_have_url(re.compile(rf"#/m/{meetings['main']}$"))
        expect(page.get_by_role("heading", level=1)).to_have_text("E2E quarter planning")

        # both note sets are offered; the local one is free, the Claude one shows its cost
        qwen = page.get_by_role("button", name=re.compile("Notes by Qwen 2.5 7B"))
        expect(qwen).to_contain_text("6 notes, 1 to check, free, written on this computer")
        expect(page.get_by_role("button", name=re.compile("Notes by Sonnet 5.5"))).to_contain_text("$0.21")
        qwen.click()
        expect(qwen).to_have_attribute("aria-pressed", "true")
        expect(page.get_by_text(QWEN_SUMMARY)).to_be_visible()
        expect(page.locator("li").filter(has_text="Beta ships on March 3")).to_be_visible()
        expect(page.get_by_role("button", name="Check this")).to_be_visible()
        # notes are grouped by topic: a short story, what was decided, questions and answers
        topic = page.get_by_role("region", name=re.compile("^Quarter plan"))
        expect(topic.get_by_text("The quarter plan is set.")).to_be_visible()
        expect(topic.get_by_role("heading", name="Questions asked")).to_be_visible()
        expect(topic.locator("li").filter(has_text="When does the beta ship?")).to_contain_text("On March 3.")
        expect(page.get_by_role("region", name=re.compile("^Still open"))).to_contain_text(
            "Who owns the pricing page?")
        transcript = page.locator("section[aria-labelledby=transcript-h]")
        expect(transcript).to_contain_text("Okay, we'll ship the beta on March 3.")
        expect(transcript).to_contain_text("Priya")

        # where notes are written, and only this server's models are offered
        where = page.get_by_test_id("notes-where")
        offers = _write_offers(page)
        if config["notes_llm"] == "local":
            expect(where).to_have_text(re.compile(r"^Notes: .+ on this computer$"))
            assert not any(re.search("Opus|Sonnet|Claude", o) for o in offers), offers
        else:
            expect(where).to_have_text(re.compile(r"^Notes: Claude"))
            assert any("Opus 5.5" in o for o in offers), offers

        # clicking a cited note moves the player to that line (u0002 starts at 0:12)
        page.wait_for_function("document.querySelector('audio')?.readyState >= 1")
        note = page.locator("li").filter(has_text="Beta ships on March 3")
        note.get_by_role("button", name="Play from 0:12").click()
        page.wait_for_function("Math.abs(document.querySelector('audio').currentTime - 12) < 2")
        expect(transcript.locator('li[data-seq="2"]')).to_have_class(re.compile("cite-flash"))
        page.evaluate("document.querySelector('audio').pause()")
        _shot(page, "meeting-1440.png")

        page.get_by_role("tab", name=re.compile("Speakers")).click()
        expect(page.get_by_text("Names come from voices you asked the app to remember")).to_be_visible()
        expect(page.get_by_role("tab", name=re.compile("Speakers"))).to_contain_text("1/2 named")
        assert tab.errors == []


def test_local_mode_never_offers_claude(browser, server, meetings):
    with Tab(browser, server, 1440, LOCAL_CONFIG) as tab:
        page = tab.page
        page.goto(f"/#/m/{meetings['opus']}")
        expect(page.get_by_text(OPUS_SUMMARY)).to_be_visible()
        expect(page.get_by_role("button", name=re.compile("Notes by Opus 5.5"))).to_contain_text("$0.41")
        offer = page.get_by_role("button", name=re.compile("Also write notes with Qwen 2.5 7B"))
        expect(offer).to_contain_text("free, on this computer")
        offers = _write_offers(page)
        assert len(offers) == 1 and not re.search("Opus|Sonnet|Claude", offers[0]), offers
        expect(page.get_by_test_id("notes-where")).to_have_text("Notes: Qwen 2.5 7B on this computer")

        page.goto(f"/#/m/{meetings['main']}")   # already has Qwen notes: nothing to offer
        expect(page.get_by_text(QWEN_SUMMARY)).to_be_visible()   # the server's model is shown first
        assert _write_offers(page) == []

        page.goto(f"/#/m/{meetings['busy']}")
        expect(page.get_by_text("About 10–15 minutes per hour of audio")).to_be_visible()
        expect(page.get_by_text("Qwen 2.5 7B on this computer", exact=True)).to_be_visible()
        assert tab.errors == []


def test_claude_mode_offers_the_missing_claude_model(browser, server, meetings):
    with Tab(browser, server, 1440, CLAUDE_CONFIG) as tab:
        page = tab.page
        page.goto(f"/#/m/{meetings['main']}")   # has Sonnet + Qwen notes
        expect(page.get_by_role("button", name=re.compile("Notes by Sonnet 5.5"))).to_be_visible()
        offers = _write_offers(page)
        assert len(offers) == 1 and offers[0].startswith("Also write notes with Opus 5.5"), offers
        expect(page.get_by_test_id("notes-where")).to_have_text("Notes: Claude (Opus 5.5)")

        page.goto(f"/#/m/{meetings['opus']}")
        expect(page.get_by_text(OPUS_SUMMARY)).to_be_visible()
        offers = _write_offers(page)
        assert len(offers) == 1 and offers[0].startswith("Also write notes with Sonnet 5.5"), offers

        page.goto(f"/#/m/{meetings['busy']}")
        expect(page.get_by_text("About 4–7 minutes per hour of audio")).to_be_visible()
        assert tab.errors == []


def test_config_unavailable_still_shows_the_meeting(browser, server, meetings):
    with Tab(browser, server, 1440, 503) as tab:
        page = tab.page
        page.goto(f"/#/m/{meetings['main']}")
        expect(page.get_by_role("button", name=re.compile("Notes by Qwen 2.5 7B"))).to_be_visible()
        expect(page.locator("li").filter(has_text="Beta ships on March 3")).to_be_visible()
        assert _write_offers(page) == []
        expect(page.get_by_test_id("notes-where")).to_have_count(0)
        # the only error is the failed /api/config request itself
        assert [e for e in tab.errors if "503" not in e] == []


def test_record_a_call_page(browser, server):
    with Tab(browser, server, 1440) as tab:
        page = tab.page
        page.goto("/#/record")
        expect(page.get_by_role("tab", name="Record a call")).to_have_attribute("aria-selected", "true")
        form = page.get_by_role("form", name="Record a call")
        expect(form).to_be_visible()
        expect(form.get_by_role("button", name="Start recording")).to_be_visible()
        expect(form).to_contain_text("My team knows our calls are recorded")
        assert tab.errors == []


def test_phone_width_has_no_horizontal_scroll(browser, server, meetings):
    with Tab(browser, server, 390) as tab:
        page = tab.page
        page.goto(f"/#/m/{meetings['main']}")
        page.get_by_role("button", name=re.compile("Notes by Qwen 2.5 7B")).click()
        expect(page.get_by_text(QWEN_SUMMARY)).to_be_visible()
        expect(page.locator("section[aria-labelledby=transcript-h]")).to_be_visible()
        _no_horizontal_scroll(page, 390)
        _shot(page, "meeting-390.png")

        page.get_by_role("tab", name=re.compile("Speakers")).click()
        _no_horizontal_scroll(page, 390)
        page.get_by_role("tab", name="Ask").click()
        expect(page.get_by_label("Your question")).to_be_visible()
        _no_horizontal_scroll(page, 390)

        page.goto("/#/record")
        expect(page.get_by_role("form", name="Record a call")).to_be_visible()
        expect(page.get_by_role("link", name=re.compile("E2E quarter planning"))).to_be_visible()
        _no_horizontal_scroll(page, 390)
        assert tab.errors == []


def test_ask_about_the_meeting(browser, server, meetings):
    """The Ask tab sends the question (and earlier turns), shows the cited answer, and a
    citation plays that moment. The answer is canned: no LLM is called."""
    sent = []

    def answer(route):
        body = json.loads(route.request.post_data)
        sent.append(body)
        first = not body["history"]
        route.fulfill(json={
            "found": first, "model": "claude-sonnet-5-5", "cost_usd": 0.045 if first else 0.004,
            "parts": [{"text": "SPEAKER_01 said the marketing budget is forty thousand dollars.",
                       "utterance_ids": ["u0001"]}] if first else
                     [{"text": "Hiring wasn't discussed.", "utterance_ids": []}]})

    with Tab(browser, server, 1440) as tab:
        page = tab.page
        page.route(f"**/api/meetings/{meetings['main']}/ask", answer)
        page.goto(f"/#/m/{meetings['main']}")
        page.get_by_role("tab", name="Ask").click()
        ask = page.get_by_role("region", name="Ask about this meeting")
        ask.get_by_label("Your question").fill("What's the marketing budget?")
        ask.get_by_label("Your question").press("Enter")
        expect(ask).to_contain_text("Speaker 2 said the marketing budget is forty thousand dollars.")
        expect(ask).to_contain_text("From the transcript, Sonnet 5.5, $0.045")
        ask.get_by_role("button", name="Play from 0:04").click()
        transcript = page.locator("section[aria-labelledby=transcript-h]")
        expect(transcript.locator('li[data-seq="1"]')).to_have_class(re.compile("cite-flash"))

        ask.get_by_label("Your question").fill("And hiring?")
        ask.get_by_role("button", name="Ask", exact=True).click()
        expect(ask).to_contain_text("Not found in this meeting")
        assert sent[1]["history"] == [{"question": "What's the marketing budget?",
                                       "answer": "SPEAKER_01 said the marketing budget is forty thousand dollars."}]
        # the conversation survives switching tabs
        page.get_by_role("tab", name="Notes").click()
        page.get_by_role("tab", name="Ask").click()
        expect(ask).to_contain_text("And hiring?")
        assert tab.errors == []
