"""Meeting Q&A (pipeline/ask.py) without an LLM: retrieval, citation checks, follow-ups."""

from pipeline import ask, llm

ROWS = [{"seq": i, "speaker_label": f"SPEAKER_0{i % 2}", "start_ms": i * 5000, "end_ms": i * 5000 + 4000,
         "text": t} for i, t in enumerate(
    ["Good morning everyone."] + [f"Filler line number {n} about the roadmap." for n in range(30)]
    + ["The marketing budget is forty thousand dollars.", "Okay, that works for me."]
    + [f"More filler {n} on design reviews." for n in range(30)])]
SPEAKERS = [{"cluster_label": "SPEAKER_00", "display_name": "Priya"},
            {"cluster_label": "SPEAKER_01", "display_name": None}]


def _fake(found=True, parts=(("Forty thousand dollars.", ["u0031", "u9999"]),)):
    calls = []

    def parse(schema, *, model, system, messages, **kw):
        calls.append({"model": model, "system": system, "messages": messages})
        return schema(found=found, parts=[{"text": t, "utterance_ids": ids} for t, ids in parts]), \
            {"cost_usd": 0.01}
    return parse, calls


def test_retrieval_finds_the_line_that_answers():
    picked = ask._retrieve(ask._utterances(ROWS), "what is the marketing budget")
    assert "u0031" in {u["id"] for u in picked}
    assert len(picked) < len(ROWS)


def test_claude_reads_whole_meeting_cached_and_bad_citations_dropped(monkeypatch):
    parse, calls = _fake()
    monkeypatch.setattr(llm, "PROVIDER", "claude")
    monkeypatch.setattr(llm, "parse", parse)
    r = ask.ask("m1", "Planning", ROWS, SPEAKERS, None, "What's the budget?",
                [{"question": "Who spoke first?", "answer": "SPEAKER_00."}])
    assert r["parts"] == [{"text": "Forty thousand dollars.", "utterance_ids": ["u0031"]}]
    system = calls[0]["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "u0062" in system[1]["text"] and "SPEAKER_00 = Priya" in system[1]["text"]
    assert [m["role"] for m in calls[0]["messages"]] == ["user", "assistant", "user"]
    assert calls[0]["model"] == llm.CLAUDE_VERIFIER


def test_local_cites_only_lines_it_was_shown(monkeypatch):
    parse, calls = _fake(parts=(("Forty thousand.", ["u0031", "u0060"]),))
    monkeypatch.setattr(llm, "PROVIDER", "local")
    monkeypatch.setattr(llm, "parse", parse)
    r = ask.ask("m1", "Planning", ROWS, SPEAKERS, None, "what is the marketing budget", [])
    shown = calls[0]["messages"][-1]["content"]
    assert "u0031" in shown and "u0060" not in shown
    assert r["parts"][0]["utterance_ids"] == ["u0031"]


def test_not_found_passes_through(monkeypatch):
    parse, _ = _fake(found=False, parts=(("That wasn't discussed.", []),))
    monkeypatch.setattr(llm, "PROVIDER", "claude")
    monkeypatch.setattr(llm, "parse", parse)
    r = ask.ask("m1", "Planning", ROWS, SPEAKERS, None, "Hiring?", [])
    assert r["found"] is False and r["parts"][0]["utterance_ids"] == []
