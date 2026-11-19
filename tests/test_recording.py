"""Desktop recording: echo suppression, mixing and "which speaker is you" on synthetic audio."""

import numpy as np
import soundfile as sf

from pipeline import recording as rec

SR = rec.SR
RNG = np.random.default_rng(7)


def burst_signal(spans, seconds, level_db=-22.0):
    """Speech-like noise: syllable-rate amplitude modulated bursts inside the given spans."""
    x = np.zeros(int(seconds * SR), np.float32)
    for a, b in spans:
        n = int((b - a) * SR)
        t = np.arange(n) / SR
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + RNG.uniform(0, 6))
        x[int(a * SR):int(a * SR) + n] = RNG.normal(size=n) * env
    rms = np.sqrt(np.mean(x[x != 0] ** 2))
    return (x / rms * 10 ** (level_db / 20)).astype(np.float32)


def room_echo(x, delay_ms=50, level_db=-10.0):
    """What a laptop mic hears from its speakers: delayed, quieter, with a short room tail."""
    ir = RNG.normal(size=int(0.15 * SR)) * np.exp(-np.arange(int(0.15 * SR)) / (0.03 * SR))
    ir /= np.sqrt(np.sum(ir**2))
    y = np.convolve(x, ir)[:len(x)] * 10 ** (level_db / 20)
    d = int(delay_ms * SR / 1000)
    return np.concatenate([np.zeros(d, np.float32), y[:-d]]).astype(np.float32)


def write(path, x):
    path.write_bytes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


YOU = [(1, 6), (14, 19), (27, 33)]
THEM = [(6.5, 13), (19.5, 26.5), (33.5, 39)]
SECONDS = 40


def scenario(tmp_path, headphones: bool, others_offset_s=0.1):
    you = burst_signal(YOU, SECONDS)
    them = burst_signal(THEM, SECONDS, -18.0)
    mic = you + RNG.normal(size=len(you)).astype(np.float32) * 10 ** (-65 / 20)
    if not headphones:
        mic = mic + room_echo(them)
    off = int(others_offset_s * SR)
    write(tmp_path / "mic-0.pcm", mic)
    write(tmp_path / "others-0.pcm", them[off:])
    return [rec.Track("mic", tmp_path / "mic-0.pcm", 0),
            rec.Track("others", tmp_path / "others-0.pcm", off)]


def frames(spans):
    return np.concatenate([np.arange(int(a * SR / rec.FRAME) + 5, int(b * SR / rec.FRAME) - 5)
                           for a, b in spans])


def turns():
    return [(a, b, "SPEAKER_01") for a, b in YOU] + [(a, b, "SPEAKER_00") for a, b in THEM]


def test_echo_is_suppressed_without_headphones_and_your_voice_kept(tmp_path):
    tracks = scenario(tmp_path, headphones=False)
    mic_db, oth_db = rec.levels(tracks)
    echo = rec.estimate_echo(mic_db, oth_db)
    assert 1 <= echo["lag"] <= 5                      # 50 ms speaker->mic delay ~ 2-3 frames
    g = rec.mic_gains(mic_db, oth_db, echo)
    assert np.mean(g[frames(THEM)] < 1) > 0.9         # echo-only frames turned down
    assert np.mean(g[frames(YOU)] == 1) > 0.95        # you are never turned down


def test_mix_writes_aligned_audio_and_levels(tmp_path):
    tracks = scenario(tmp_path, headphones=False)
    info = rec.mix(tracks, tmp_path / "out.wav", tmp_path / "levels.npz")
    wav, sr = sf.read(tmp_path / "out.wav", dtype="float32")
    assert sr == SR and len(wav) == SECONDS * SR and info["duration_s"] == SECONDS
    assert info["limited_samples"] == 0 and info["aec"]["removed_db_median"] > 10
    # remote speech is in the mix at full level; your speech too
    lvl = rec.frame_db(wav)
    assert np.median(lvl[frames(THEM)]) > -30 and np.median(lvl[frames(YOU)]) > -30
    saved = np.load(tmp_path / "levels.npz")
    assert len(saved["mic_db"]) == len(lvl)


def test_your_cluster_is_found_with_and_without_headphones(tmp_path):
    for headphones in (True, False):
        d = tmp_path / str(headphones)
        d.mkdir()
        mic_db, oth_db = rec.levels(scenario(d, headphones))
        echo = rec.estimate_echo(mic_db, oth_db)
        *_, mic_h, oth_h = rec.balance(mic_db, oth_db, echo)
        shares = rec.owner_shares(turns(), mic_h, oth_h, echo["lag"])
        assert shares["SPEAKER_01"]["share"] > 0.8 and shares["SPEAKER_00"]["share"] < 0.2
        assert rec.owner_label(shares)[0] == "SPEAKER_01"


def test_nobody_is_called_you_in_an_in_person_meeting(tmp_path):
    """Everyone talks into the one laptop mic: two clusters are mic-only, so abstain."""
    room = burst_signal(YOU, SECONDS) + burst_signal(THEM, SECONDS)
    write(tmp_path / "mic-0.pcm", room)
    write(tmp_path / "others-0.pcm", np.zeros(SECONDS * SR, np.float32))
    tracks = [rec.Track("mic", tmp_path / "mic-0.pcm", 0),
              rec.Track("others", tmp_path / "others-0.pcm", 0)]
    mic_db, oth_db = rec.levels(tracks)
    assert rec.estimate_echo(mic_db, oth_db)["coupling_db"] is None
    assert rec.owner_label(rec.owner_shares(turns(), mic_db, oth_db))[0] is None
    rec.mix(tracks, tmp_path / "out.wav")              # no loopback audio: still mixes


def test_tracks_from_a_device_switch_are_summed_on_one_clock(tmp_path):
    a = np.full(SR, 0.1, np.float32)
    write(tmp_path / "a.pcm", a)
    write(tmp_path / "b.pcm", a)
    tracks = [rec.Track("others", tmp_path / "a.pcm", 0),
              rec.Track("others", tmp_path / "b.pcm", 2 * SR)]   # headphones plugged in at 2 s
    assert rec.total_length(tracks) == 3 * SR
    x = rec.read(tracks, "others", 0, 3 * SR)
    assert np.allclose(x[:SR], 0.1, atol=1e-3) and np.all(x[SR:2 * SR] == 0)
    assert np.allclose(x[2 * SR:], 0.1, atol=1e-3)


def test_headphones_leave_the_mic_untouched(tmp_path):
    mic_db, oth_db = rec.levels(scenario(tmp_path, headphones=True))
    echo = rec.estimate_echo(mic_db, oth_db)
    assert echo["coupling_db"] is None and echo["lag"] == 0
    assert np.all(rec.mic_gains(mic_db, oth_db, echo) == 1)


def test_you_are_found_even_when_the_call_carries_your_voice_too(tmp_path):
    """Someone in the room with you (or AMI's shared room) puts your voice, quieter, into the
    call audio as well. The mic is still clearly louder for you."""
    you = burst_signal(YOU, SECONDS)
    them = burst_signal(THEM, SECONDS, -18.0)
    mic = you + room_echo(them, level_db=-20) + RNG.normal(size=len(you)).astype(np.float32) * 1e-3
    call = them + room_echo(you, delay_ms=5, level_db=-18)        # your voice leaks into the call
    write(tmp_path / "mic-0.pcm", mic)
    write(tmp_path / "others-0.pcm", call)
    tracks = [rec.Track("mic", tmp_path / "mic-0.pcm", 0),
              rec.Track("others", tmp_path / "others-0.pcm", 0)]
    rec.mix(tracks, tmp_path / "out.wav", tmp_path / "lv.npz")
    label, ev = rec.owner_from_levels(turns(), np.load(tmp_path / "lv.npz"))
    assert label == "SPEAKER_01" and ev["share"] > 0.8 and ev["next_share"] < 0.2


def test_echo_canceller_removes_echo_and_keeps_you_during_double_talk():
    from pipeline.aec import EchoCanceller

    you_spans = [(14, 19), (30, 33), (45, 50)]
    them_spans = [(2, 13), (19.5, 26.5), (28, 39), (41, 50)]       # 30-33, 45-50: both talk
    you = burst_signal(you_spans, 55, -30.0)
    them = burst_signal(them_spans, 55, -20.0)
    echo = room_echo(them, delay_ms=60, level_db=-10.0)             # echo as loud as you
    ac = EchoCanceller(-10.0)
    out = np.concatenate([ac.process((you + echo)[i:i + SR * 10], them[i:i + SR * 10])
                          for i in range(0, 55 * SR, SR * 10)] + [ac.flush()])

    def db(x):
        return 10 * np.log10(np.mean(x**2) + 1e-20)

    sl = slice(20 * SR, 26 * SR)                                     # echo only, converged
    assert db(echo[sl]) - db(out[sl]) > 15
    for a, b in [(30, 33), (45, 50)]:                                # double talk
        sl = slice(a * SR, b * SR)
        assert db(you[sl]) - db(out[sl] - you[sl]) > 6               # was 0 dB before
