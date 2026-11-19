"""Local LLM through Ollama: same contract as the Claude path in pipeline/llm.py, $0.

Structured output uses Ollama's `format` = the Pydantic model's JSON schema (constrained
decoding), then the reply is validated with Pydantic; a reply that still doesn't validate is
sent back once with the validation error. Temperature 0 and a fixed seed, so a run repeats.

GPU: the model shares an 8 GB card with Whisper and pyannote. The pipeline frees those before
the notes stage, and `unload()` releases this model when a meeting's notes are done, so the
next meeting's transcription has the memory back.

Calls are logged to artifacts/llm_local.jsonl (tokens and time; cost is always 0), not to the
spend ledger, which therefore stays unchanged while the local model is in use.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "llm_local.jsonl"
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
MODEL = os.environ.get("LOCAL_LLM_MODEL", "qwen2.5:7b")
NUM_CTX = int(os.environ.get("LOCAL_LLM_NUM_CTX", "12288"))   # one size: a change reloads
MAX_CTX = int(os.environ.get("LOCAL_LLM_MAX_CTX", "32768"))   # qwen2.5 supports 32k
KEEP_ALIVE = os.environ.get("LOCAL_LLM_KEEP_ALIVE", "10m")
SEED = 7

T = TypeVar("T", bound=BaseModel)
_lock = threading.Lock()


class LocalLLMError(RuntimeError):
    """Ollama unreachable or erroring: the stage fails (and is retried on the next run)."""


class LocalLLMInvalidOutput(LocalLLMError):
    """The model answered but the reply never validated: safe to skip this one call."""


def _text(content) -> str:
    """Claude-style content (a string, or a list of {"type": "text", ...} blocks) -> text."""
    if isinstance(content, str):
        return content
    return "\n\n".join(b.get("text", "") for b in content if isinstance(b, dict))


def _ctx(chars: int, max_tokens: int) -> int:
    """Context window for this call. Always NUM_CTX unless the prompt (~3.2 chars/token) plus
    reply needs more: Ollama reloads the model whenever num_ctx changes (~10 s on this GPU),
    so a fixed size avoids reloading between calls. Measured: at 24k the runner held ~15 GB
    of committed RAM on this 16 GB laptop and the system ran out of memory; 12k is plenty for
    the windowed calls (~4-7k tokens), and only the topic segmentation of a long meeting grows."""
    need = int(chars / 3.2) + max_tokens + 512
    return NUM_CTX if need <= NUM_CTX else min(MAX_CTX, -(-need // 4096) * 4096)


_http: httpx.Client | None = None


def _client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=httpx.Timeout(30, read=1800))
    return _http


def available(timeout: float = 3.0) -> bool:
    try:
        r = httpx.get(f"{HOST}/api/tags", timeout=timeout)
        return r.status_code == 200 and any(
            m.get("name") == MODEL or m.get("model") == MODEL for m in r.json().get("models", []))
    except httpx.HTTPError:
        return False


def unload(model: str = MODEL) -> None:
    """Free the model's GPU memory now (instead of after KEEP_ALIVE)."""
    try:
        httpx.post(f"{HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
    except httpx.HTTPError:
        pass


def _log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def parse(output_format: type[T], *, model: str, system: str | list, messages: list,
          purpose: str, meeting: str | None = None, max_tokens: int = 8000,
          run: str | None = None, client: httpx.Client | None = None) -> tuple[T, dict]:
    """One structured call. Returns (parsed object, usage record with cost_usd = 0)."""
    msgs = [{"role": "system", "content": _text(system)}]
    msgs += [{"role": m["role"], "content": _text(m["content"])} for m in messages]
    schema = output_format.model_json_schema()
    chars = sum(len(m["content"]) for m in msgs)
    if int(chars / 3.2) + max_tokens + 512 > MAX_CTX:
        raise LocalLLMError(f"{purpose}: prompt of ~{int(chars / 3.2)} tokens does not fit the "
                            f"local model's {MAX_CTX}-token context")
    num_ctx = _ctx(chars, max_tokens)
    http = client or _client()
    t0 = time.perf_counter()
    prompt_tokens = out_tokens = 0
    last_err = None
    for attempt in range(2):
        body = {"model": model, "messages": msgs, "stream": False, "format": schema,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": 0, "seed": SEED, "num_ctx": num_ctx,
                            "num_predict": max_tokens}}
        try:
            r = http.post(f"{HOST}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LocalLLMError(f"{purpose}: Ollama is not reachable at {HOST} ({e}). "
                                "Start Ollama, or set NOTES_LLM=claude.") from e
        if r.status_code != 200:
            raise LocalLLMError(f"{purpose}: Ollama error {r.status_code}: {r.text[:300]}")
        d = r.json()
        prompt_tokens += d.get("prompt_eval_count", 0) or 0
        out_tokens += d.get("eval_count", 0) or 0
        content = d.get("message", {}).get("content", "")
        try:
            parsed = output_format.model_validate_json(content)
            break
        except ValidationError as e:
            last_err = e
            if d.get("done_reason") == "length":
                raise LocalLLMInvalidOutput(f"{purpose}: hit max_tokens={max_tokens}") from e
            msgs = msgs + [{"role": "assistant", "content": content},
                           {"role": "user", "content": "That JSON does not match the schema: "
                            f"{str(e)[:800]}\nReturn the corrected JSON only."}]
            num_ctx = _ctx(sum(len(m["content"]) for m in msgs), max_tokens)
    else:
        raise LocalLLMInvalidOutput(f"{purpose}: no valid JSON after retry ({last_err})")
    rec = {"ts": datetime.now(UTC).isoformat(), "provider": "local", "run": run,
           "purpose": purpose, "meeting": meeting, "model": model, "num_ctx": num_ctx,
           "attempts": attempt + 1, "input_tokens": prompt_tokens, "output_tokens": out_tokens,
           "elapsed_s": round(time.perf_counter() - t0, 2), "cost_usd": 0.0}
    _log(rec)
    return parsed, rec
