"""The one door to an LLM: structured outputs, spend ledger, budget guard, provider switch.

NOTES_LLM picks the provider (in .env; default "local"):
  local   Qwen through Ollama on this machine (pipeline/llm_local.py). $0. Claude is then
          BLOCKED: any Claude call raises ClaudeDisabled before a request is built, so no
          money can be spent by accident. The spend ledger is not touched.
  claude  Claude Opus 5.5 writes the notes, Sonnet 5.5 verifies (the original setup).
          Every call is appended to artifacts/llm_spend.jsonl with tokens and cost, and
          refused before it is sent if the budget scope's running total would exceed the cap.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel

import pipeline  # noqa: F401  (loads .env)

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "artifacts" / "llm_spend.jsonl"

CLAUDE_EXTRACTOR = "claude-opus-5-5"
CLAUDE_VERIFIER = "claude-sonnet-5-5"
PROVIDER = os.environ.get("NOTES_LLM", "local").strip().lower()
if PROVIDER not in ("local", "claude"):
    raise ValueError(f"NOTES_LLM must be 'local' or 'claude', not {PROVIDER!r}")
LOCAL_MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:7b")
# the models used when a caller doesn't name one
EXTRACTOR = CLAUDE_EXTRACTOR if PROVIDER == "claude" else LOCAL_MODEL
VERIFIER = CLAUDE_VERIFIER if PROVIDER == "claude" else LOCAL_MODEL

# $ per 1M tokens: input, output, cache read. Cache writes (5 min) bill at 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}

BUDGET_SCOPE = os.environ.get("LLM_BUDGET_SCOPE", "step1")
BUDGET_USD = float(os.environ.get("LLM_BUDGET_USD", "15"))

T = TypeVar("T", bound=BaseModel)
_lock = threading.Lock()
_client: anthropic.Anthropic | None = None
RUN_TAG: str | None = None  # set by eval runners so concurrent runs can report their own spend


class BudgetExceeded(RuntimeError):
    pass


class ClaudeDisabled(RuntimeError):
    """A Claude call while NOTES_LLM=local: refused before anything is sent."""


def is_local(model: str) -> bool:
    return not model.startswith("claude-")


def stage_suffix(model: str) -> str:
    """'' for the original Claude models, else a filesystem-safe tag like 'qwen2.5-7b'."""
    return model.removeprefix("claude-").replace(":", "-").replace("/", "-")


def client() -> anthropic.Anthropic:
    global _client
    if PROVIDER != "claude":
        raise ClaudeDisabled("Claude is switched off (NOTES_LLM=local); set NOTES_LLM=claude "
                             "in .env and restart to use it")
    if _client is None:
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def cost_usd(model: str, usage) -> float:
    p_in, p_out, p_cache = PRICES[model]
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (usage.input_tokens * p_in + cache_write * p_in * 1.25 + cache_read * p_cache
            + usage.output_tokens * p_out) / 1e6


def spent(scope: str = BUDGET_SCOPE, run: str | None = None, meeting: str | None = None) -> float:
    """Total spend in a budget scope, optionally only for one run tag and/or meeting."""
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        rec = json.loads(line)
        if (rec.get("scope") == scope and (run is None or rec.get("run") == run)
                and (meeting is None or rec.get("meeting") == meeting)):
            total += rec["cost_usd"]
    return total


def _record(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LEDGER.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def parse(output_format: type[T], *, model: str, system: str | list, messages: list,
          purpose: str, meeting: str | None = None, effort: str = "medium",
          max_tokens: int = 16000) -> tuple[T, dict]:
    """One structured-output call. Returns (parsed object, usage record)."""
    if is_local(model):
        from pipeline import llm_local

        return llm_local.parse(output_format, model=model, system=system, messages=messages,
                               purpose=purpose, meeting=meeting, run=RUN_TAG,
                               max_tokens=min(max_tokens, 4000))
    if PROVIDER != "claude":
        raise ClaudeDisabled(f"{purpose}: {model} requested but Claude is switched off "
                             "(NOTES_LLM=local)")
    with _lock:
        total = spent()
    if total >= BUDGET_USD:
        raise BudgetExceeded(f"{BUDGET_SCOPE} spend ${total:.2f} >= cap ${BUDGET_USD:.2f}")

    # Always stream: long transcripts and large max_tokens exceed the SDK's non-streaming limit.
    with client().beta.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_format=output_format,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        msg = stream.get_final_message()
    rec = {
        "ts": datetime.now(UTC).isoformat(), "scope": BUDGET_SCOPE, "run": RUN_TAG,
        "purpose": purpose, "meeting": meeting, "model": msg.model, "effort": effort,
        "stop_reason": msg.stop_reason,
        "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
        "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        "cost_usd": round(cost_usd(model, msg.usage), 5),
    }
    _record(rec)
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"{purpose}: refused ({msg.stop_details})")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError(f"{purpose}: hit max_tokens={max_tokens}")
    if msg.parsed_output is None:
        raise RuntimeError(f"{purpose}: no parsed output (stop_reason={msg.stop_reason})")
    return msg.parsed_output, rec
