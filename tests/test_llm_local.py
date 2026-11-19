"""Local LLM provider and the $0 guard: no GPU, no network (a fake Ollama answers)."""

import json

import httpx
import pytest
from pydantic import BaseModel

from pipeline import llm, llm_local


class Note(BaseModel):
    text: str
    ids: list[str]


def fake_ollama(replies: list[str], seen: list[dict]):
    """An httpx client whose /api/chat returns the given message contents in turn."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"message": {"content": replies[len(seen) - 1]},
                                         "prompt_eval_count": 50, "eval_count": 10,
                                         "done_reason": "stop"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_structured_call_sends_the_schema_and_costs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_local, "LOG", tmp_path / "local.jsonl")
    seen: list[dict] = []
    out, rec = llm_local.parse(
        Note, model="qwen2.5:7b", purpose="t",
        system=[{"type": "text", "text": "sys"}],
        messages=[{"role": "user", "content": [{"type": "text", "text": "a"},
                                               {"type": "text", "text": "b"}]}],
        client=fake_ollama(['{"text": "Ship Friday", "ids": ["u1"]}'], seen))
    assert out == Note(text="Ship Friday", ids=["u1"]) and rec["cost_usd"] == 0.0
    body = seen[0]
    assert body["format"] == Note.model_json_schema()          # constrained decoding
    assert body["options"]["temperature"] == 0 and body["options"]["seed"] == llm_local.SEED
    assert [m["content"] for m in body["messages"]] == ["sys", "a\n\nb"]   # blocks -> text
    assert (tmp_path / "local.jsonl").exists()


def test_invalid_json_is_sent_back_once_with_the_error(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_local, "LOG", tmp_path / "local.jsonl")
    seen: list[dict] = []
    out, rec = llm_local.parse(Note, model="qwen2.5:7b", purpose="t", system="s",
                               messages=[{"role": "user", "content": "x"}],
                               client=fake_ollama(['{"text": "no ids"}',
                                                   '{"text": "ok", "ids": []}'], seen))
    assert out.text == "ok" and rec["attempts"] == 2
    assert "does not match the schema" in seen[1]["messages"][-1]["content"]


def test_a_reply_that_never_validates_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_local, "LOG", tmp_path / "local.jsonl")
    with pytest.raises(llm_local.LocalLLMError, match="no valid JSON"):
        llm_local.parse(Note, model="qwen2.5:7b", purpose="t", system="s",
                        messages=[{"role": "user", "content": "x"}],
                        client=fake_ollama(["nope", "still nope"], []))


def test_context_window_fits_the_prompt():
    assert llm_local._ctx(1000, 2000) == llm_local.NUM_CTX          # one size: no reloads
    assert llm_local._ctx(20000, 4000) == llm_local.NUM_CTX
    assert llm_local._ctx(80000, 4000) == 32768                    # only if it must grow
    assert llm_local._ctx(10**6, 8000) == llm_local.MAX_CTX


def test_claude_is_blocked_when_the_local_provider_is_active(monkeypatch):
    """G0: with NOTES_LLM=local no Claude request can be built, so no money can be spent."""
    monkeypatch.setattr(llm, "PROVIDER", "local")
    with pytest.raises(llm.ClaudeDisabled):
        llm.client()
    with pytest.raises(llm.ClaudeDisabled):
        llm.parse(Note, model="claude-opus-5-5", system="s", messages=[], purpose="t")


def test_stage_names_keep_claude_and_local_results_apart():
    from pipeline.run import insights_stage

    assert insights_stage("claude-opus-5-5") == "insights"          # cached Opus notes kept
    assert insights_stage("claude-sonnet-5-5") == "insights-sonnet-5-5"
    assert insights_stage("qwen2.5:7b") == "insights-qwen2.5-7b"


def test_ollama_being_down_fails_loudly_instead_of_skipping(tmp_path, monkeypatch):
    """A transport error must fail the notes stage (so it is retried), not be skipped like a
    bad reply, which would silently drop part of the notes."""
    monkeypatch.setattr(llm_local, "LOG", tmp_path / "local.jsonl")

    def down(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(llm_local.LocalLLMError) as e:
        llm_local.parse(Note, model="qwen2.5:7b", purpose="t", system="s",
                        messages=[{"role": "user", "content": "x"}],
                        client=httpx.Client(transport=httpx.MockTransport(down)))
    assert not isinstance(e.value, llm_local.LocalLLMInvalidOutput)


def test_a_prompt_too_long_for_the_context_is_refused_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_local, "LOG", tmp_path / "local.jsonl")
    with pytest.raises(llm_local.LocalLLMError, match="does not fit"):
        llm_local.parse(Note, model="qwen2.5:7b", purpose="t", system="s",
                        messages=[{"role": "user", "content": "x" * 200_000}],
                        client=fake_ollama(['{"text": "", "ids": []}'], []))
