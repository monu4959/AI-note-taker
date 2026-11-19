"""Step 2: turn a desktop recording into one meeting audio file, and find which speaker is you.

A recording has two kinds of track, 16 kHz mono int16, placed on one clock by sample offset:

  mic     your microphone: you, plus (without headphones) the call coming out of your
          speakers a few tens of ms later and quieter, i.e. echo
  others  what the computer played (loopback of every output device): everyone else

The pipeline takes one file, so the tracks are mixed. Mixed naively, the echo makes every
remote word appear twice, ~50 ms apart, and once the mic is levelled up to your voice the
echo can be louder than the original. Two stages remove it:
  1. echo cancellation (pipeline/aec.py): an adaptive filter learns speaker -> room -> mic
     from the loopback and subtracts the predicted echo, keeping your voice when you talk
     over the call (~20 dB removed);
  2. residual suppression: frames where the mic still carries nothing but echo are turned
     down. "Nothing but echo" = the mic is no louder than the measured speaker-to-mic
     coupling predicts from what was being played.
Both run only when echo is detected (the mic follows the loopback); with headphones nothing
is touched.

The two tracks also tell us who you are: your voice is much louder on the mic than on the
loopback; everyone else is the other way round (after echo removal and with both tracks at
the same speech level). "Louder", not "the loopback is silent": someone in the room with you
can carry your voice into the call too. `owner_label` picks the one diarization cluster
whose speech is mostly mic-only, or nobody (never guess, e.g. an in-person meeting where
every voice comes through the mic).

Everything is chunked: an hour is 57.6 M samples per track, too much to hold several float
copies of next to the GPU worker on a 16 GB laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
FRAME = 320                 # 20 ms analysis frame
CHUNK = SR * 60             # samples per processing chunk (a multiple of FRAME)
SILENT_DB = -100.0
ACTIVE_DB = -50.0           # a loopback frame above this is "the call is playing"
MAX_ECHO_LAG = 15           # frames (300 ms): speaker -> air -> mic delay searched
MIN_ECHO_CORRELATION = 0.4   # below this the mic does not follow the loopback: headphones
ECHO_MARGIN_DB = 6.0        # mic within 6 dB of predicted echo -> echo only
SUPPRESS_GAIN = 10 ** (-24 / 20)
KEEP_HOLD = 3               # frames of speech kept around each kept frame (no chopped onsets)
TARGET_DB = -20.0           # each track's speech level before mixing
MIX_VERSION = 3             # bump when the mix changes (cached eval mixes are rebuilt)
LIMIT = 0.9                 # soft limiter knee (levelled tracks summed can exceed 0 dBFS)
MAX_TRACK_GAIN_DB = 30.0
DOMINANT_DB = 10.0          # a mic frame is "yours" when 10 dB above the loopback (levelled)
OWNER_MIN_SHARE = 0.35       # your cluster: at least this share of mic-dominant speech ...
OWNER_MARGIN = 3.0           # ... and this many times the next cluster's share
OWNER_MIN_SPEECH_S = 10.0


@dataclass
class Track:
    kind: str               # "mic" | "others"
    path: Path              # raw int16 little-endian mono at SR
    offset: int             # first sample's position on the recording clock

    @property
    def length(self) -> int:
        return self.path.stat().st_size // 2 if self.path.exists() else 0


def total_length(tracks: list[Track]) -> int:
    return max((t.offset + t.length for t in tracks), default=0)


def read(tracks: list[Track], kind: str, a: int, b: int) -> np.ndarray:
    """Samples [a, b) of all tracks of one kind, summed (devices change mid-meeting)."""
    out = np.zeros(b - a, np.float32)
    for t in tracks:
        if t.kind != kind or t.length == 0:
            continue
        lo, hi = max(a, t.offset), min(b, t.offset + t.length)
        if lo >= hi:
            continue
        data = np.memmap(t.path, dtype="<i2", mode="r")
        out[lo - a:hi - a] += data[lo - t.offset:hi - t.offset].astype(np.float32) / 32768.0
        del data
    return out


def frame_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0, np.float32)
    f = x[:n * FRAME].reshape(n, FRAME).astype(np.float64)
    rms = np.sqrt(np.mean(f * f, axis=1))
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-10)), SILENT_DB).astype(np.float32)


def levels(tracks: list[Track]) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame dBFS of the mic and the others track over the whole recording."""
    n = total_length(tracks)
    mic, oth = [], []
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        mic.append(frame_db(read(tracks, "mic", a, b)))
        oth.append(frame_db(read(tracks, "others", a, b)))
    def cat(xs):
        return np.concatenate(xs) if xs else np.zeros(0, np.float32)
    return cat(mic), cat(oth)


def _shift(x: np.ndarray, lag: int, fill: float) -> np.ndarray:
    """x delayed by `lag` frames."""
    if lag <= 0:
        return x
    return np.concatenate([np.full(lag, fill, x.dtype), x[:-lag]])


def _max_filter(x: np.ndarray, r: int) -> np.ndarray:
    if r <= 0 or len(x) == 0:
        return x
    p = np.pad(x, r, mode="edge")
    return np.max(np.lib.stride_tricks.sliding_window_view(p, 2 * r + 1), axis=1)


def estimate_echo(mic_db: np.ndarray, oth_db: np.ndarray, lag: int | None = None,
                  min_correlation: float | None = None) -> dict:
    """How the call leaks into the mic: delay (frames) and coupling (dB, mic minus loopback).

    Lag = the delay at which mic and loopback envelopes correlate best while the call plays;
    coupling = median mic-minus-loopback level at that lag. The median is taken over frames
    where the call plays, which are mostly not you talking, so your speech barely moves it.
    """
    playing = oth_db > ACTIVE_DB
    if playing.sum() < 50:          # < 1 s of call audio: in-person meeting, nothing to do
        return {"lag": 0, "coupling_db": None, "correlation": None}
    best = (-2.0, 0)
    for cand in ([lag] if lag is not None else range(MAX_ECHO_LAG + 1)):
        o = _shift(oth_db, cand, SILENT_DB)
        sel = _shift(playing, cand, False)
        if sel.sum() < 50:
            continue
        a, b = mic_db[sel], o[sel]
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best[0]:
            best = (c, cand)
    corr, lag = best
    if min_correlation is None:
        min_correlation = MIN_ECHO_CORRELATION
    if corr < min_correlation:   # headphones: no echo to remove, leave the mic untouched
        return {"lag": 0, "coupling_db": None, "correlation": round(corr, 3)}
    sel = _shift(playing, lag, False)
    coupling = float(np.median(mic_db[sel] - _shift(oth_db, lag, SILENT_DB)[sel]))
    return {"lag": lag, "coupling_db": round(coupling, 2), "correlation": round(corr, 3)}


def mic_gains(mic_db: np.ndarray, oth_db: np.ndarray, echo: dict) -> np.ndarray:
    """Per-frame gain for the mic: 1 where it carries you, SUPPRESS_GAIN where only echo."""
    g = np.ones(len(mic_db), np.float32)
    if echo["coupling_db"] is None:
        return g
    # what the echo alone would measure on the mic, allowing for a little timing smear
    predicted = _max_filter(_shift(oth_db, echo["lag"], SILENT_DB), 2) + echo["coupling_db"]
    playing = _max_filter(_shift(oth_db, echo["lag"], SILENT_DB), 2) > ACTIVE_DB
    keep = ~(playing & (mic_db <= predicted + ECHO_MARGIN_DB))
    keep = _max_filter(keep.astype(np.int8), KEEP_HOLD).astype(bool)
    g[~keep] = SUPPRESS_GAIN
    return g


def speech_gain(db: np.ndarray) -> float:
    """Linear gain that brings a track's active frames to TARGET_DB (0 for an empty track)."""
    floor = np.percentile(db, 10) if len(db) else SILENT_DB
    active = db[(db > max(ACTIVE_DB, floor + 10))]
    if len(active) < 50:   # (almost) nothing on this track: leave it as recorded
        return 1.0
    level = 10 * np.log10(np.mean(10 ** (active.astype(np.float64) / 10)))
    return float(10 ** (min(TARGET_DB - level, MAX_TRACK_GAIN_DB) / 20))


def balance(mic_db: np.ndarray, oth_db: np.ndarray, echo: dict):
    """Mic gains (echo removal), track gains, and the levels as they will be heard in the mix:
    (g, gm, go, mic_heard_db, oth_heard_db)."""
    g = mic_gains(mic_db, oth_db, echo)
    mic_after = mic_db + 20 * np.log10(np.maximum(g, 1e-10))
    gm, go = speech_gain(mic_after), speech_gain(oth_db)
    mic_heard = mic_after + 20 * np.log10(max(gm, 1e-10))
    oth_heard = oth_db + 20 * np.log10(max(go, 1e-10))
    return g, gm, go, mic_heard.astype(np.float32), oth_heard.astype(np.float32)


def cancel_echo(tracks: list[Track], echo: dict, dst: Path) -> list[Track]:
    """Run the echo canceller over the whole mic; returns the tracks with the mic replaced
    by one echo-cancelled track on the recording clock."""
    from pipeline.aec import EchoCanceller

    n = total_length(tracks)
    # pre-delay the reference by the measured lag (minus 40 ms margin) so the 256 ms filter
    # spends its taps on the room tail, not on the delay
    pre = max(0, echo["lag"] * FRAME - 640)
    ac = EchoCanceller(echo["coupling_db"])
    with dst.open("wb") as fh:
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            e = ac.process(read(tracks, "mic", a, b), read(tracks, "others", a - pre, b - pre))
            fh.write((np.clip(e, -1, 1) * 32767).astype("<i2").tobytes())
        fh.write((np.clip(ac.flush(), -1, 1) * 32767).astype("<i2").tobytes())
    return [t for t in tracks if t.kind != "mic"] + [Track("mic", dst, 0)]


def soft_limit(y: np.ndarray) -> np.ndarray:
    """Leave everything below LIMIT alone; squash the rest smoothly below 1.0."""
    a = np.abs(y)
    over = a > LIMIT
    if over.any():
        room = 0.99 - LIMIT
        y = y.copy()
        y[over] = np.sign(y[over]) * (LIMIT + room * np.tanh((a[over] - LIMIT) / room))
    return y


def mix(tracks: list[Track], dst: Path, levels_path: Path | None = None) -> dict:
    """Write the meeting audio (16 kHz mono PCM16) and the per-frame levels. Returns info."""
    mic_db, oth_db = levels(tracks)
    echo = estimate_echo(mic_db, oth_db)
    aec = None
    if echo["coupling_db"] is not None:          # no headphones: cancel, then re-measure
        dst.parent.mkdir(parents=True, exist_ok=True)
        tracks = cancel_echo(tracks, echo, dst.parent / "mic-aec.pcm")
        raw_mic_db = mic_db
        mic_db, _ = levels(tracks)
        playing = oth_db[:len(mic_db)] > ACTIVE_DB
        drop = (raw_mic_db[:len(mic_db)] - mic_db)[playing]
        aec = {"removed_db_median": round(float(np.median(drop)), 1) if len(drop) else None}
        # what is left still comes from the speakers (we know there was echo), even if it no
        # longer follows the loopback closely: suppress it at the same delay
        residual = estimate_echo(mic_db, oth_db, lag=echo["lag"], min_correlation=-1.0)
        echo = {**echo, "residual": residual}
        echo_for_gate = residual
    else:
        echo_for_gate = echo
    # loudness balance from what will actually be heard (the mic after suppression)
    g, gm, go, mic_heard, oth_heard = balance(mic_db, oth_db, echo_for_gate)
    n = total_length(tracks)
    frame_centres = np.arange(len(g)) * FRAME + FRAME / 2
    dst.parent.mkdir(parents=True, exist_ok=True)
    clipped = 0
    with sf.SoundFile(dst, "w", SR, 1, subtype="PCM_16") as out:
        for a in range(0, n, CHUNK):
            b = min(n, a + CHUNK)
            ramp = np.interp(np.arange(a, b), frame_centres, g, left=1.0, right=1.0)
            y = read(tracks, "mic", a, b) * ramp * gm + read(tracks, "others", a, b) * go
            clipped += int(np.sum(np.abs(y) > LIMIT))
            out.write(np.clip(soft_limit(y), -0.99, 0.99).astype(np.float32))
    if levels_path is not None:
        np.savez_compressed(levels_path, mic_db=mic_db, oth_db=oth_db, lag=echo["lag"],
                            mic_heard_db=mic_heard, oth_heard_db=oth_heard)
    for t in tracks:
        if t.path.name == "mic-aec.pcm":
            t.path.unlink(missing_ok=True)
    suppressed = float(np.mean(g < 1)) if len(g) else 0.0
    return {"version": MIX_VERSION, "duration_s": round(n / SR, 3), "echo": echo, "aec": aec,
            "mic_suppressed_share": round(suppressed, 4),
            "mic_gain_db": round(float(20 * np.log10(max(gm, 1e-10))), 1),
            "others_gain_db": round(float(20 * np.log10(max(go, 1e-10))), 1),
            "limited_samples": clipped,
            "tracks": {k: sum(t.length for t in tracks if t.kind == k) / SR
                       for k in ("mic", "others")}}


def owner_shares(turns: list[tuple[float, float, str]], mic_db: np.ndarray,
                 oth_db: np.ndarray, lag: int = 0) -> dict[str, dict]:
    """For each diarization cluster: how much of its speech is dominated by the mic.

    Pass the levels as heard in the mix (`balance`): echo removed, both tracks at the same
    speech level. A frame is the cluster's if inside one of its (exclusive) turns. "Mic
    dominant" = the mic is clearly above its noise floor and DOMINANT_DB above the loopback
    (shifted by the echo delay, widened a little). Your cluster scores near 1, remote
    speakers near 0.
    """
    n = len(mic_db)
    if n == 0:
        return {}
    floor = float(np.percentile(mic_db, 10))
    mic_on = mic_db > max(floor + 15, ACTIVE_DB)
    oth = _max_filter(_shift(oth_db[:n], lag, SILENT_DB), 3)
    oth_on = oth > ACTIVE_DB
    mine = mic_on & (mic_db > oth + DOMINANT_DB)
    out: dict[str, dict] = {}
    for start, end, label in turns:
        a, b = int(start * SR / FRAME), min(n, int(end * SR / FRAME))
        if b <= a:
            continue
        d = out.setdefault(label, {"mic_only": 0, "voiced": 0, "speech_s": 0.0})
        d["mic_only"] += int(np.sum(mine[a:b]))
        d["voiced"] += int(np.sum(mic_on[a:b] | oth_on[a:b]))
        d["speech_s"] += end - start
    for d in out.values():
        d["share"] = round(d["mic_only"] / d["voiced"], 3) if d["voiced"] else 0.0
        d["speech_s"] = round(d["speech_s"], 1)
    return out


def owner_from_levels(turns, lv) -> tuple[str | None, dict]:
    """Owner cluster from a saved levels file (older files have only the raw levels)."""
    if "mic_heard_db" in lv:
        shares = owner_shares(turns, lv["mic_heard_db"], lv["oth_heard_db"], int(lv["lag"]))
    else:
        shares = owner_shares(turns, lv["mic_db"], lv["oth_db"], int(lv["lag"]))
    return owner_label(shares)


def owner_label(shares: dict[str, dict]) -> tuple[str | None, dict]:
    """The one cluster that is you, or None when it is not clear-cut.

    Relative, not absolute: measured on AMI (everyone in one room, so the call track carries
    your voice too) your share was 0.45-0.58 while every other cluster stayed <= 0.08. So the
    top cluster is you when it has OWNER_MIN_SHARE and OWNER_MARGIN times the runner-up.
    In an in-person meeting every voice is on the mic, shares are all high: abstain.
    """
    ranked = sorted(((lab, d) for lab, d in shares.items()
                     if d["speech_s"] >= OWNER_MIN_SPEECH_S), key=lambda kv: -kv[1]["share"])
    if not ranked:
        return None, {"share": None, "next_share": None}
    top, d = ranked[0]
    nxt = ranked[1][1]["share"] if len(ranked) > 1 else 0.0
    evidence = {"share": d["share"], "next_share": nxt if len(ranked) > 1 else None}
    if d["share"] >= OWNER_MIN_SHARE and d["share"] >= OWNER_MARGIN * max(nxt, 0.05):
        return top, evidence
    return None, evidence
