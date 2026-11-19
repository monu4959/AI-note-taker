from pyannote.core import Annotation, Segment

from pipeline.assign import assign_words, to_utterances


def _turns():
    ann = Annotation()
    ann[Segment(0.0, 2.0)] = "A"
    ann[Segment(2.0, 4.0)] = "B"
    ann[Segment(10.0, 12.0)] = "A"
    return ann


def _w(start, end, word):
    return {"start": start, "end": end, "word": word, "prob": 0.9}


def test_word_goes_to_max_overlap_turn():
    (w,) = assign_words([_w(1.5, 2.8, " hi")], _turns())  # 0.5 s in A, 0.8 s in B
    assert w["speaker"] == "B" and not w["far"]


def test_gap_word_goes_to_nearest_turn_and_flags_far():
    near, far = assign_words([_w(4.2, 4.4, " a"), _w(6.0, 6.5, " b")], _turns())
    assert near["speaker"] == "B" and not near["far"]  # 0.2 s from B
    assert far["speaker"] == "B" and far["far"]  # 2.0 s from B, 3.5 s from A


def test_utterances_split_on_speaker_change_and_long_pause():
    words = [_w(0.1, 0.5, " one"), _w(0.6, 1.0, " two"), _w(2.5, 3.0, " three"),
             _w(10.2, 10.6, " four"), _w(11.8, 12.0, " five")]
    utts = to_utterances(assign_words(words, _turns()))
    assert [(u["id"], u["speaker"], u["text"]) for u in utts] == [
        ("u0000", "A", "one two"),
        ("u0001", "B", "three"),
        ("u0002", "A", "four"),  # pause 1.2 s > 1.0 s splits the A run
        ("u0003", "A", "five"),
    ]
