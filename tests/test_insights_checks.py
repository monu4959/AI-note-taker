from types import SimpleNamespace as NS

from pipeline.insights.extract import _fix_spans
from pipeline.insights.verify import deterministic_checks

UTTS = {
    "u0001": {"id": "u0001", "speaker": "SPEAKER_00",
              "text": "We closed Q3 with churn at four point two percent, down from 5.1.",
              "words": [{"word": " four", "prob": 0.95}, {"word": " two", "prob": 0.4}]},
    "u0002": {"id": "u0002", "speaker": "SPEAKER_01", "text": "Thanks Priya, I'll send it Friday.",
              "words": []},
}
SPEAKERS = {"SPEAKER_00", "SPEAKER_01"}
LOWER = " ".join(u["text"] for u in UTTS.values()).lower()


def _check(item, name):
    return next(c for c in deterministic_checks(item, UTTS, SPEAKERS, LOWER) if c["check"] == name)


def _metric(value):
    return {"type": "metric", "name": "Q3 churn", "value": value, "unit": "%",
            "evidence": {"utterance_ids": ["u0001"]}}


def test_metric_value_matches_spoken_number_words():
    assert _check(_metric("4.2"), "metric_value_in_text")["ok"]


def test_metric_value_not_in_text_fails_hard():
    c = _check(_metric("42"), "metric_value_in_text")
    assert not c["ok"] and c["hard"]


def test_low_confidence_number_word_is_flagged_soft():
    c = _check(_metric("4.2"), "number_confidence")
    assert not c["ok"] and not c["hard"]


def test_missing_citation_fails_hard():
    c = _check({"type": "decision", "text": "x", "evidence": {"utterance_ids": ["u9999"]}},
               "ids_exist")
    assert not c["ok"] and c["hard"]


def test_owner_must_be_speaker_or_name_said():
    def action(owner):
        return {"type": "action", "task": "send deck", "owner": owner, "due": None,
                "evidence": {"utterance_ids": ["u0002"]}}
    assert _check(action("SPEAKER_01"), "owner_known")["ok"]
    assert _check(action("Priya"), "owner_known")["ok"]
    assert not _check(action("Raj"), "owner_known")["ok"]


def test_fix_spans_closes_gaps_overlaps_and_drops_unknown_ids():
    ids = [f"u{i:04d}" for i in range(10)]
    spans = [NS(title="B", start_id="u0004", end_id="u0007"),
             NS(title="A", start_id="u0001", end_id="u0005"),   # overlaps B, misses u0000
             NS(title="X", start_id="u0042", end_id="u0050")]   # unknown IDs
    assert _fix_spans(spans, ids) == [
        {"title": "A", "start_id": "u0000", "end_id": "u0003"},
        {"title": "B", "start_id": "u0004", "end_id": "u0009"},
    ]


def test_a_weak_local_judge_flags_but_never_verifies_or_deletes(monkeypatch):
    from pipeline import llm
    from pipeline.insights import verify as v

    utts = [{"id": "u0001", "speaker": "SPEAKER_00", "text": "We ship on Friday.", "words": []}]
    item = {"id": "d1", "type": "decision", "text": "Ship on Friday",
            "evidence": {"utterance_ids": ["u0001"]}}
    monkeypatch.setattr(v, "semantic_verdicts",
                        lambda items, u, m: {"d1": {"support": "unsupported", "reason": "x"}})
    monkeypatch.setattr(llm, "VERIFIER", "qwen2.5:7b")
    assert v.verify([dict(item)], utts, "m")[0]["verification"]["status"] == "needs_review"
    monkeypatch.setattr(llm, "VERIFIER", "claude-sonnet-5-5")
    assert v.verify([dict(item)], utts, "m")[0]["verification"]["status"] == "rejected"


def test_topic_spans_survive_duplicates_at_the_end_and_empty_meetings():
    from types import SimpleNamespace as S

    from pipeline.insights.extract import _fix_spans

    ids = [f"u{i:04d}" for i in range(30)]
    spans = [S(title="a", start_id="u0000", end_id="u0010")] + \
        [S(title=f"x{i}", start_id="u0029", end_id="u0029") for i in range(4)]
    out = _fix_spans(spans, ids)
    assert out[0]["start_id"] == "u0000" and out[-1]["end_id"] == "u0029"
    assert _fix_spans([], []) == []
