"""Local notes judge and its calibration against Sonnet: agreement math, pre-filter, batching,
file naming. No GPU, no network: the LLM is faked."""

import json
import sys
from pathlib import Path

import pytest

from evaluation import insights_eval, local_judge
from evaluation import judge_agreement as ja
from pipeline import llm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step4_calibrate_judge as step4

# --- agreement math ---------------------------------------------------------------------


def test_kappa_perfect_chance_and_known_value():
    assert ja.cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # classic 2x2 example: po = 0.7, pe = 0.5 -> kappa 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert ja.cohen_kappa(a, b) == pytest.approx(0.4)
    # agreement no better than chance
    assert ja.cohen_kappa(["y", "n", "y", "n"], ["y", "y", "n", "n"]) == pytest.approx(0.0)


def test_kappa_undefined_cases():
    assert ja.cohen_kappa([], []) is None
    assert ja.cohen_kappa(["found"] * 3, ["found"] * 3) is None      # pe = 1
    with pytest.raises(ValueError):
        ja.cohen_kappa(["a"], [])


def test_views_and_agreement():
    s = ["covered", "partial", "missed", "covered"]
    lo = ["partial", "partial", "missed", "covered"]
    assert [ja.collapse(x, "lenient") for x in s] == ["found", "found", "missed", "found"]
    assert [ja.collapse(x, "strict") for x in s] == ["covered", "not", "not", "covered"]
    rep = ja.agreement(s, lo)
    assert rep["n"] == 4
    assert rep["3way"]["agreement"] == 0.75
    assert rep["lenient"]["agreement"] == 1.0 and rep["lenient"]["kappa"] == 1.0
    assert rep["strict"]["agreement"] == 0.75
    assert rep["confusion"]["covered"] == {"covered": 1, "partial": 1, "missed": 0}


def test_skipped_sentence_counts_as_missed_and_recall():
    matches = [{"human_id": "h1", "status": "covered"}, {"human_id": "h3", "status": "partial"}]
    assert ja.statuses(matches, ["h1", "h2", "h3"]) == ["covered", "missed", "partial"]
    r = ja.recall({"decisions": ["covered", "partial", "missed", "covered"], "actions": []})
    assert r["decisions"] == {"n": 4, "strict": 0.5, "lenient": 0.75}
    assert r["actions"]["strict"] is None and r["problems"]["n"] == 0


# --- pre-filter and batching ------------------------------------------------------------


def _item(i, text, t="key_point", status="verified"):
    return {"id": i, "type": t, "text": text, "verification": {"status": status},
            "evidence": {"utterance_ids": []}}


def test_tokens_and_bm25_rank_the_matching_note_first():
    assert local_judge.tokens("The Buttons were designed") == ["button", "design"]
    docs = ["Budget is twelve euros", "Remote will have a docking station", "Lunch break"]
    assert local_judge.bm25_rank("The remote will have a docking station.", docs)[0] == 1
    assert local_judge.bm25_rank("nothing shared", docs) == [0, 1, 2]   # ties: doc order


def test_candidates_keep_note_order_and_size():
    items = [_item(f"k{i}", f"note about topic {i}") for i in range(10)]
    items[7]["text"] = "docking station for the remote"
    got = local_judge.candidates("a docking station", items, 3)
    assert len(got) == 3 and "k7" in [it["id"] for it in got]
    assert [it["id"] for it in got] == sorted((it["id"] for it in got), key=lambda x: int(x[1:]))
    assert local_judge.candidates("x", items, None) == items


def test_batches():
    assert local_judge.batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert local_judge.batches([1, 2, 3], None) == [[1, 2, 3]]
    assert local_judge.batches([], 3) == []


class FakeLLM:
    """Stands in for llm.parse: answers from a function of the user message."""
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, output_format, *, model, system, messages, purpose, meeting=None,
                 effort="medium", max_tokens=16000):
        self.calls.append({"format": output_format, "model": model,
                           "user": messages[0]["content"]})
        return output_format.model_validate(self.answer(messages[0]["content"])), {}


INSIGHTS = {"items": [_item("d1", "The remote will have a docking station", "decision"),
                      _item("k1", "Budget is twelve euros fifty"),
                      _item("k2", "Rejected note", status="rejected")]}
HUMAN = [{"id": "h1", "section": "decisions", "text": "The remote will have a docking station."},
         {"id": "h2", "section": "problems", "text": "Whether to include a scroll wheel."}]


def test_focused_judge_one_call_per_sentence_and_cleans_output(monkeypatch):
    def answer(user):
        if "docking" in user.split("SYSTEM NOTES")[0]:
            return {"item_ids": ["d1", "zz9"], "reason": "same", "status": "covered"}
        return {"item_ids": ["k1"], "reason": "not there", "status": "missed"}
    fake = FakeLLM(answer)
    monkeypatch.setattr(llm, "parse", fake)
    monkeypatch.setitem(local_judge.LOCAL_JUDGES, 99, {"batch": 1, "top_k": 1,
                                                       "prompt": "focused"})
    out = local_judge.judge_local("M", INSIGHTS, HUMAN, "t", "qwen2.5:7b", version=99)
    assert len(fake.calls) == 2 and all(c["model"] == "qwen2.5:7b" for c in fake.calls)
    assert "Rejected note" not in fake.calls[0]["user"]
    assert fake.calls[0]["user"].count("\nd1:") + fake.calls[0]["user"].count("\nk1:") == 1
    assert out == [
        {"human_id": "h1", "status": "covered", "item_ids": ["d1"], "reason": "same"},
        {"human_id": "h2", "status": "missed", "item_ids": [], "reason": "not there"}]


def test_batched_judge_fills_a_skipped_sentence_as_missed(monkeypatch):
    fake = FakeLLM(lambda user: {"matches": [
        {"human_id": "h1", "item_ids": ["d1"], "reason": "r", "status": "partial"}]})
    monkeypatch.setattr(llm, "parse", fake)
    monkeypatch.setitem(local_judge.LOCAL_JUDGES, 98, {"batch": 2, "top_k": None,
                                                       "prompt": "focused"})
    out = local_judge.judge_local("M", INSIGHTS, HUMAN, "t", "qwen2.5:7b", version=98)
    assert len(fake.calls) == 1
    assert [(m["human_id"], m["status"]) for m in out] == [("h1", "partial"), ("h2", "missed")]


def test_local_judge_refuses_a_claude_model():
    with pytest.raises(ValueError):
        local_judge.judge_local("M", INSIGHTS, HUMAN, "t", "claude-sonnet-5-5")


def test_judge_routes_a_local_model_to_the_local_judge(monkeypatch):
    seen = {}
    monkeypatch.setattr(local_judge, "judge_local",
                        lambda m, i, h, t, model, version: seen.update(v=version, m=model) or [])
    insights_eval.judge("M", INSIGHTS, HUMAN, "t", model="qwen2.5:7b")
    assert seen == {"v": local_judge.LOCAL_JUDGE_VERSION, "m": "qwen2.5:7b"}


# --- file naming and pair discovery -----------------------------------------------------


def test_judge_filenames_keep_claude_names_and_tag_local_ones():
    v = local_judge.LOCAL_JUDGE_VERSION
    assert insights_eval.judge_filename("abc123", "claude-sonnet-5-5") == "judge.j1.abc123.json"
    assert insights_eval.judge_filename("ref-p2", "claude-sonnet-5-5") == "judge.j1.ref-p2.json"
    assert (insights_eval.judge_filename("abc123", "qwen2.5:7b")
            == f"judge.j{v}-qwen2.5-7b.abc123.json")
    assert insights_eval.judge_filename("abc", "qwen2.5:7b", 1) == "judge.j1-qwen2.5-7b.abc.json"
    assert insights_eval.judge_version("claude-sonnet-5-5") == insights_eval.JUDGE_VERSION


def test_find_pairs_matches_sonnet_files_only(tmp_path):
    a = tmp_path / "asr" / "M1"
    r = tmp_path / "ref" / "M2"
    a.mkdir(parents=True)
    r.mkdir(parents=True)
    for p in (a / "insights.f00.json", a / "judge.j1.f00.json", a / "judge.j1.gone.json",
              a / "judge.j1-qwen2.5-7b.f00.json", r / "insights.p2.json",
              r / "judge.j1.ref-p2.json"):
        p.write_text(json.dumps([]))
    pairs = step4.find_pairs(tmp_path)
    assert [(p["source"], p["meeting"], p["cache_id"], p["insights"].name) for p in pairs] == [
        ("asr", "M1", "f00", "insights.f00.json"), ("ref", "M2", "ref-p2", "insights.p2.json")]
    path = step4.local_path(pairs[0], "qwen2.5:7b", 2)
    assert path.name == "judge.j2-qwen2.5-7b.f00.json" and path != pairs[0]["sonnet"]
    with pytest.raises(RuntimeError):
        step4.local_path(pairs[0], "claude-sonnet-5-5", 1)


def test_compare_pools_agreement_recall_and_shared_items():
    human = [{"id": "h1", "section": "decisions", "text": ""},
             {"id": "h2", "section": "actions", "text": ""}]
    row = {"human": human, "items": [_item("d1", "x", "decision"), _item("a1", "y", "action")],
           "sonnet": [{"human_id": "h1", "status": "covered", "item_ids": ["d1"]},
                      {"human_id": "h2", "status": "missed", "item_ids": []}],
           "local": [{"human_id": "h1", "status": "partial", "item_ids": ["d1"]},
                     {"human_id": "h2", "status": "partial", "item_ids": ["a1"]}]}
    rep = step4.compare([row])
    assert rep["sentences"] == 2 and rep["agreement"]["3way"]["agreement"] == 0.0
    assert rep["recall"]["sonnet"]["decisions"] == {"n": 1, "strict": 1.0, "lenient": 1.0}
    assert rep["recall"]["local"]["actions"] == {"n": 1, "strict": 0.0, "lenient": 1.0}
    assert rep["precision_lb"]["action"] == {"n": 1, "sonnet": 0.0, "local": 1.0}
    assert rep["shared_item_rate"] == {"both_found": 1, "value": 1.0}


def test_status_from_elements():
    f = local_judge.status_from_elements
    assert f(["stated", "stated"]) == "covered"
    assert f(["stated", "absent"]) == "partial"
    assert f(["stated", "contradicted"]) == "partial"
    assert f(["absent", "contradicted"]) == "missed"
    assert f([]) == "missed"


def test_elements_judge_derives_status_and_ids(monkeypatch):
    fake = FakeLLM(lambda user: {"elements": [
        {"element": "docking station", "note_ids": ["d1"], "verdict": "stated"},
        {"element": "who", "note_ids": ["k1"], "verdict": "absent"}]})
    monkeypatch.setattr(llm, "parse", fake)
    monkeypatch.setitem(local_judge.LOCAL_JUDGES, 97, {"batch": 1, "top_k": None,
                                                       "prompt": "elements"})
    out = local_judge.judge_local("M", INSIGHTS, HUMAN[:1], "t", "qwen2.5:7b", version=97)
    assert out[0]["status"] == "partial" and out[0]["item_ids"] == ["d1"]
    monkeypatch.setitem(local_judge.LOCAL_JUDGES, 96, {"batch": 2, "top_k": None,
                                                       "prompt": "elements"})
    with pytest.raises(ValueError):
        local_judge.judge_local("M", INSIGHTS, HUMAN, "t", "qwen2.5:7b", version=96)
