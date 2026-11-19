import numpy as np

from pipeline import name_cues, speaker_id
from pipeline.name_cues import NameCue
from pipeline.speaker_id import Cluster


def _vec(*xs):
    v = np.zeros(256)
    v[: len(xs)] = xs
    return speaker_id.normalise(v)


GALLERY = {"ana": [_vec(1, 0, 0)], "raj": [_vec(0, 1, 0)], "li": [_vec(0, 0, 1)]}


def test_identify_is_one_to_one_and_thresholded():
    clusters = [Cluster("S0", _vec(1, 0.1, 0), 60), Cluster("S1", _vec(0.9, 0.2, 0), 60),
                Cluster("S2", _vec(0.3, 0.3, 0.3), 60)]
    got = {m.label: m.person for m in speaker_id.identify(clusters, GALLERY, threshold=0.8)}
    # S0 and S1 both look like ana; only the better one gets the name, S1 gets nobody >= 0.8
    assert got == {"S0": "ana"}


def test_identify_skips_short_clusters_and_respects_candidates():
    clusters = [Cluster("S0", _vec(1, 0, 0), 5), Cluster("S1", _vec(0, 1, 0), 60)]
    assert [m.label for m in speaker_id.identify(clusters, GALLERY, threshold=0.5)] == ["S1"]
    assert speaker_id.identify(clusters, GALLERY, threshold=0.5, candidates={"ana"}) == []


def test_person_score_uses_best_voiceprints():
    g = {"ana": [_vec(1, 0, 0), _vec(0, 0, 1)]}   # two very different recordings
    people, S = speaker_id.score_matrix([Cluster("S0", _vec(1, 0, 0), 60)], g, top_k=1)
    assert people == ["ana"] and abs(S[0, 0] - 1.0) < 1e-9


U = [
    {"id": "u0000", "speaker": "S0", "text": "Hi everyone, I'm Priya and I run design."},
    {"id": "u0001", "speaker": "S1", "text": "Thanks. Raj, can you share the numbers?"},
    {"id": "u0002", "speaker": "S2", "text": "Sure, here they are."},
    {"id": "u0003", "speaker": "S1", "text": "As Maya said last week, churn is up."},
]


def cue(speaker, name, kind, ids, spoken=None):
    return NameCue(speaker=speaker, name=name, spoken_as=spoken or name, kind=kind,
                   utterance_ids=ids)


def test_cue_checks():
    assert name_cues.check(cue("S0", "Priya", "self_introduction", ["u0000"]), U) is None
    assert name_cues.check(cue("S1", "Priya", "self_introduction", ["u0000"]), U) is not None
    assert name_cues.check(cue("S2", "Raj", "addressed", ["u0001"]), U) is None
    # Maya is only talked about; nobody named Maya replies next to it
    assert name_cues.check(cue("S0", "Maya", "addressed", ["u0003"]), U) is not None
    assert name_cues.check(cue("S0", "Priyanka", "self_introduction", ["u0000"]), U) is not None
    assert name_cues.check(cue("S9", "Priya", "self_introduction", ["u0000"]), U) is not None


def test_resolve_abstains_on_conflicts():
    cues = [cue("S0", "Priya", "self_introduction", ["u0000"]),
            cue("S2", "Raj", "addressed", ["u0001"]),
            cue("S1", "Raj", "addressed", ["u0001"])]   # rejected: S1 said it
    r = name_cues.resolve(cues, U)
    assert {k: v["name"] for k, v in r["accepted"].items()} == {"S0": "Priya", "S2": "Raj"}

    # the same name supported for two speakers: neither gets it
    u = U + [{"id": "u0004", "speaker": "S0", "text": "Right, Raj."},
             {"id": "u0005", "speaker": "S1", "text": "Yes?"}]
    r = name_cues.resolve([cue("S2", "Raj", "addressed", ["u0001"]),
                           cue("S1", "Raj", "addressed", ["u0004"])], u)
    assert r["accepted"] == {}
