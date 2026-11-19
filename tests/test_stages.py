from pipeline.stages import MeetingStore, run_stage


def _pipeline(store, calls, asr_model="a", diarize_output="d"):
    def stage(name, content=None):
        def fn(out):
            calls.append(name)
            (out / "x.txt").write_text(content or name)
        return fn

    k1 = run_stage(store, "prepare", 1, {}, {}, stage("prepare"))
    k2 = run_stage(store, "transcribe", 1, {"model": asr_model}, {"prepare": k1},
                   stage("transcribe", f"words from {asr_model}"))
    k3 = run_stage(store, "diarize", 1, {}, {"prepare": k1}, stage("diarize", diarize_output))
    run_stage(store, "assign", 1, {}, {"transcribe": k2, "diarize": k3}, stage("assign"))


def test_rerun_is_fully_cached(tmp_path):
    store, calls = MeetingStore(tmp_path, "m1"), []
    _pipeline(store, calls)
    _pipeline(store, calls)
    assert calls == ["prepare", "transcribe", "diarize", "assign"]


def test_param_change_reruns_stage_and_downstream_only(tmp_path):
    store, calls = MeetingStore(tmp_path, "m1"), []
    _pipeline(store, calls)
    calls.clear()
    _pipeline(store, calls, asr_model="b")
    assert calls == ["transcribe", "assign"]  # prepare and diarize stay cached


def test_unfinished_stage_reruns_and_identical_output_keeps_downstream_cached(tmp_path):
    store, calls = MeetingStore(tmp_path, "m1"), []
    _pipeline(store, calls)
    (store.dir("diarize") / "manifest.json").unlink()  # simulate a crash mid-stage
    calls.clear()
    _pipeline(store, calls)
    assert calls == ["diarize"]  # same output as before -> assign stays valid


def test_unfinished_stage_with_different_output_reruns_downstream(tmp_path):
    store, calls = MeetingStore(tmp_path, "m1"), []
    _pipeline(store, calls)
    (store.dir("diarize") / "manifest.json").unlink()
    calls.clear()
    _pipeline(store, calls, diarize_output="different turns")  # e.g. non-deterministic rerun
    assert calls == ["diarize", "assign"]
