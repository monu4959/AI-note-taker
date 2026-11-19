"""Step 2: record a meeting on this computer (Granola-style: no bot joins the call).

While recording, one thread per audio device writes 16 kHz mono int16 to its own file:

  mic-N.pcm      the default microphone (you). Follows Windows' default mic if it changes.
  others-N.pcm   loopback of EVERY output device (what Zoom/Teams plays: everyone else).
                 All of them, because the call may play on headphones, speakers, a monitor or
                 a Bluetooth headset, and you may plug headphones in halfway through.

Each file's first sample is placed on the recording clock by its start time (`offset`), so
devices can come and go. Windows delivers nothing from a loopback while nothing plays;
soundcard fills that with silence by the clock, and a stalled thread is padded the same way.

Clock lock: every sound device has its own crystal. Measured on this laptop the mic ran
~450 ppm fast (1.6 s per hour) and the HDMI output ~700 ppm slow, against the system clock.
So each writer keeps its track locked to the system clock: when it drifts ahead (behind) it
resamples the current 100 ms block a few samples shorter (longer). Inaudible, and the tracks
stay within ~1 ms of each other for any length of meeting.

Crash safety: audio is appended to disk every 100 ms with `meta.json` beside it. If the API
dies (or the PC reboots) mid-meeting, the folder survives and shows up as an interrupted
recording you can still turn into notes.

On stop: the meeting is created at once (the UI can open it), then a background thread
mixes the tracks (pipeline/recording.py) and hands the result to a sink:
  LocalSink   (services/recorder_local.py) this machine's database and GPU worker
  RemoteSink  (services/agent.py)          upload to the team's notetaker server
If delivery fails (server unreachable), the folder stays and shows up as interrupted, so
the recording can be sent again. This module needs no database: it also runs on a
teammate's Mac or PC as part of the recorder agent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from pipeline import recording as rec
from services import calls

log = logging.getLogger("recorder")

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "data" / "recordings"
SR = rec.SR
BLOCK = SR // 10                 # 100 ms per read
BUFFER_S = 0.5                   # WASAPI buffer: a thread may stall this long without loss
MONITOR_S = 2.0
CALL_END_GRACE_S = 30            # the call app released the mic this long ago -> call over
MAX_S = 4 * 3600
MAX_FAILURES = 3                 # a device that fails this often is left alone
LOCK_BLOCKS = 20                 # blocks used to measure a track's fixed start latency
LOCK_GAIN = 2                    # samples of drift beyond the deadband per sample adjusted
LOCK_MAX = 8                     # max samples added/removed per block (0.5 %)
LOCK_DEADBAND = 24               # samples (1.5 ms) of timing jitter left alone
SOURCE_NAME = "Recorded on this computer"


@dataclass(frozen=True)
class Device:
    id: str
    name: str
    kind: str                    # "mic" | "others"


class Sink(Protocol):
    """Where finished recordings go."""
    def create(self, meta: dict) -> str: ...                      # -> meeting id
    def exists(self, meeting_id: str) -> bool: ...
    def deliver(self, rec_dir: Path, meeting_id: str, meta: dict) -> dict: ...
    def fail(self, meeting_id: str, error: str) -> None: ...


class Backend(Protocol):
    def default_mic(self) -> Device | None: ...
    def outputs(self) -> list[Device]: ...
    def open(self, device: Device): ...   # context manager with .record(numframes) -> array


def _com_init() -> None:
    """Each capture thread talks to WASAPI (COM) and must initialise COM for itself."""
    if sys.platform == "win32":
        import ctypes

        import soundcard  # noqa: F401  (first: its import initialises COM and rejects S_FALSE)

        ctypes.windll.ole32.CoInitializeEx(None, 0)   # COINIT_MULTITHREADED; S_FALSE is fine


class SoundcardBackend:
    """Windows audio through the `soundcard` package (WASAPI, shared mode; Windows resamples
    to 16 kHz mono for us)."""

    def __init__(self):
        import soundcard

        warnings.filterwarnings("ignore", message="data discontinuity in recording")
        self.sc = soundcard

    def default_mic(self) -> Device | None:
        try:
            m = self.sc.default_microphone()
        except Exception:
            return None
        return Device(m.id, m.name, "mic")

    def outputs(self) -> list[Device]:
        return [Device(s.id, s.name, "others") for s in self.sc.all_speakers()]

    def open(self, device: Device):
        src = self.sc.get_microphone(device.id, include_loopback=device.kind == "others")
        return src.recorder(samplerate=SR, blocksize=int(SR * BUFFER_S))


class MacBackend(SoundcardBackend):
    """macOS: the mic through CoreAudio (soundcard resamples to 16 kHz with Apple's
    AudioConverter). macOS has no loopback, so the computer's sound comes from BlackHole, a
    free virtual device that the user's Multi-Output Device feeds (docs/teammate-setup.md).
    Every input whose name contains "BlackHole" counts as the call audio."""

    def __init__(self):
        import soundcard

        self.sc = soundcard

    def outputs(self) -> list[Device]:
        return [Device(m.id, m.name, "others") for m in self.sc.all_microphones()
                if "blackhole" in m.name.lower()]

    def default_mic(self) -> Device | None:
        try:
            m = self.sc.default_microphone()
        except Exception:
            return None
        if "blackhole" in m.name.lower():   # the call audio is not your mic
            real = [x for x in self.sc.all_microphones() if "blackhole" not in x.name.lower()]
            if not real:
                return None
            m = real[0]
        return Device(m.id, m.name, "mic")

    def open(self, device: Device):
        return self.sc.get_microphone(device.id).recorder(samplerate=SR,
                                                          blocksize=int(SR * 0.1))


def default_backend() -> Backend:
    return MacBackend() if sys.platform == "darwin" else SoundcardBackend()


def _db(x: np.ndarray) -> float:
    if len(x) == 0:
        return rec.SILENT_DB
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return max(20 * np.log10(max(rms, 1e-10)), rec.SILENT_DB)


class _Writer(threading.Thread):
    """Captures one device to one file until stopped (or the device disappears)."""

    def __init__(self, session: Session, device: Device, index: int):
        super().__init__(daemon=True, name=f"rec-{device.kind}-{index}")
        self.session, self.device = session, device
        self.file = f"{device.kind}-{index}.pcm"
        self.halt = threading.Event()
        self.level_db = rec.SILENT_DB
        self.error: str | None = None
        self.drift = 0.0          # samples ahead of the system clock after the lock (~0)
        self.adjusted = 0         # samples added or removed by the lock, in total

    def run(self) -> None:
        _com_init()
        s = self.session
        try:
            with s.backend.open(self.device) as reader:
                start = time.monotonic()
                s.add_track(self.file, self.device, round((start - s.t0) * SR))
                written, flushed, blocks = 0, start, 0
                first: list[float] = []
                base = ema = None
                with (s.dir / self.file).open("ab") as fh:
                    while not self.halt.is_set():
                        x = np.asarray(reader.record(numframes=BLOCK), np.float32)
                        if x.ndim == 2:
                            x = x.mean(axis=1)
                        if self.device.kind == "mic" and s.muted:
                            x = np.zeros_like(x)
                        now = time.monotonic()
                        ahead = written + len(x) - (now - start) * SR   # samples vs the clock
                        blocks += 1
                        if base is None:
                            first.append(ahead)
                            if blocks >= LOCK_BLOCKS:
                                base = float(np.median(first))
                        elif ahead - base < -SR * BUFFER_S:   # this thread stalled: pad
                            gap = int(base - ahead)
                            fh.write(np.zeros(gap, "<i2").tobytes())
                            written += gap
                            s.lost_s += gap / SR
                            ema = None
                        else:
                            ema = ahead if ema is None else 0.9 * ema + 0.1 * ahead
                            self.drift = ema - base
                            over = max(abs(self.drift) - LOCK_DEADBAND, 0.0)
                            k = int(np.sign(self.drift) * min(np.ceil(over / LOCK_GAIN), LOCK_MAX))
                            if k:                           # device clock drifted: adjust
                                self.adjusted += abs(k)
                                n = len(x) - k
                                x = np.interp(np.linspace(0, len(x) - 1, n),
                                              np.arange(len(x)), x).astype(np.float32)
                        fh.write((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
                        written += len(x)
                        self.level_db = _db(x)
                        if now - flushed > 1.0:
                            fh.flush()
                            os.fsync(fh.fileno())
                            flushed = now
        except Exception as exc:   # unplugged, or the device refused to open
            self.error = f"{type(exc).__name__}: {exc}"
            s.failures[self.device.id] = s.failures.get(self.device.id, 0) + 1
            log.warning("%s (%s) stopped: %s", self.file, self.device.name, self.error)
        finally:
            self.level_db = rec.SILENT_DB


class Session:
    """One recording in progress: its folder, its capture threads and its metadata."""

    def __init__(self, backend: Backend, root: Path, title: str, attendees: list[str],
                 owner_name: str | None, call_app: str | None):
        self.backend = backend
        self.id = uuid.uuid4().hex[:12]
        self.dir = root / self.id
        self.dir.mkdir(parents=True)
        self.t0 = time.monotonic()
        self.muted = False
        self.lost_s = 0.0
        self.writers: list[_Writer] = []
        self.failures: dict[str, int] = {}      # device id -> times its capture failed
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self.meta = {"id": self.id, "title": title, "attendees": attendees,
                     "owner_name": owner_name, "call_app": call_app,
                     "started_at": datetime.now(UTC).isoformat(), "state": "recording",
                     "tracks": []}
        self.save()

    def save(self) -> None:
        with self._lock:
            tmp = self.dir / "meta.json.tmp"
            tmp.write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
            os.replace(tmp, self.dir / "meta.json")

    def add_track(self, file: str, device: Device, offset: int) -> None:
        with self._lock:
            self.meta["tracks"].append({"file": file, "kind": device.kind,
                                        "device": device.name, "offset": offset})
        self.save()

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def live(self, kind: str) -> list[_Writer]:
        return [w for w in self.writers if w.device.kind == kind and w.is_alive()]

    def start_writer(self, device: Device) -> None:
        w = _Writer(self, device, sum(x.device.kind == device.kind for x in self.writers))
        self.writers.append(w)
        w.start()

    def halt(self, timeout: float = 3.0) -> None:
        self.stopped.set()
        for w in self.writers:
            w.halt.set()
        for w in self.writers:
            w.join(timeout)


class Recorder:
    """The one recorder of this machine. Thread-safe; the API holds a single instance."""

    def __init__(self, backend: Backend | None = None, root: Path | None = None,
                 call_detector=calls.active_calls, sink: Sink | None = None):
        self._backend = backend
        self._sink = sink
        self.root = root or RECORDINGS
        self.detect_calls = call_detector
        self.lock = threading.RLock()
        self.session: Session | None = None
        self.state = "idle"                       # idle | recording | finishing
        self.last: dict | None = None             # the most recent finished recording
        self._calls: tuple[float, list[str]] = (0.0, [])
        self._devices: tuple[float, dict | None] = (0.0, None)

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = default_backend()
        return self._backend

    @property
    def sink(self) -> Sink:
        if self._sink is None:
            from services.recorder_local import LocalSink

            self._sink = LocalSink()
        return self._sink

    # --- status --------------------------------------------------------------------------

    def active_calls(self) -> list[str]:
        t, found = self._calls
        if time.monotonic() - t > 1.5:
            try:
                found = self.detect_calls()
            except Exception:
                found = []
            self._calls = (time.monotonic(), found)
        return found

    def devices(self) -> dict | None:
        t, d = self._devices
        if d is None or time.monotonic() - t > 5:
            try:
                _com_init()
                mic = self.backend.default_mic()
                d = {"mic": mic.name if mic else None,
                     "outputs": [o.name for o in self.backend.outputs()]}
            except Exception as exc:
                d = {"mic": None, "outputs": [], "error": str(exc)}
            self._devices = (time.monotonic(), d)
        return d

    def status(self) -> dict:
        with self.lock:
            s = self.session
            out = {"state": self.state, "calls": self.active_calls(), "last": self.last,
                   "interrupted": self.interrupted(), "recording": None}
            if s is None or self.state != "recording":
                out["devices"] = self.devices()
                return out
            mics, others = s.live("mic"), s.live("others")
            warn = []
            if not mics:
                errs = [w.error for w in s.writers if w.device.kind == "mic" and w.error]
                warn.append("Your microphone isn't being recorded"
                            + (f" ({errs[-1]})" if errs else "") + ".")
            if not others:
                warn.append("The computer's sound isn't being recorded, so other people "
                            "won't be in the recording."
                            + (" Set up BlackHole (see the setup guide)."
                               if sys.platform == "darwin" else ""))
            if s.lost_s >= 1:
                warn.append(f"{s.lost_s:.0f} s of audio were lost while the computer was busy.")
            out["recording"] = {
                "id": s.id, "title": s.meta["title"], "started_at": s.meta["started_at"],
                "elapsed_s": round(s.elapsed(), 1), "muted": s.muted,
                "call_app": s.meta["call_app"], "owner_name": s.meta["owner_name"],
                "levels": {"mic": max((w.level_db for w in mics), default=rec.SILENT_DB),
                           "others": max((w.level_db for w in others), default=rec.SILENT_DB)},
                "devices": {"mic": mics[0].device.name if mics else None,
                            "outputs": [w.device.name for w in others]},
                "warnings": warn,
            }
            return out

    # --- control -------------------------------------------------------------------------

    def start(self, title: str, attendees: list[str], owner_name: str | None) -> dict:
        with self.lock:
            if self.state != "idle":
                raise RuntimeError("a recording is already in progress"
                                   if self.state == "recording" else "the last recording is still being saved")
            _com_init()
            mic = self.backend.default_mic()
            outputs = self.backend.outputs()
            if mic is None and not outputs:
                raise RuntimeError("no microphone or speakers found")
            call = (self.active_calls() or [None])[0]
            s = Session(self.backend, self.root, title, attendees, owner_name, call)
            if mic:
                s.start_writer(mic)
            for o in outputs:
                s.start_writer(o)
            self.session, self.state = s, "recording"
            threading.Thread(target=self._monitor, args=(s,), daemon=True,
                             name="rec-monitor").start()
            log.info("recording %s started (%s; call: %s)", s.id, title, call)
            return self.status()

    def set_muted(self, muted: bool) -> None:
        with self.lock:
            if self.session is None or self.state != "recording":
                raise RuntimeError("not recording")
            self.session.muted = muted

    def stop(self, reason: str = "stopped") -> str | None:
        """Stop, create the meeting at once, and mix + deliver it in the background.
        Returns None (and keeps the audio for a retry) if the meeting can't be created."""
        with self.lock:
            s = self.session
            if s is None or self.state != "recording":
                raise RuntimeError("not recording")
            s.halt()
            s.meta.update(state="stopped", ended_at=datetime.now(UTC).isoformat(),
                          stop_reason=reason, elapsed_s=round(s.elapsed(), 1))
            try:
                meeting_id = self.sink.create(s.meta)
            except Exception as exc:   # e.g. the server is unreachable: keep it for later
                log.warning("could not create the meeting for %s: %s", s.id, exc)
                s.save()
                self.session, self.state = None, "idle"
                self.last = {"meeting_id": None, "title": s.meta["title"], "reason": reason,
                             "error": f"Couldn't reach the notetaker server ({exc}). The "
                                      "audio is kept: send it again from the banner."}
                return None
            s.meta["meeting_id"] = meeting_id
            s.save()
            self.state = "finishing"
            self.last = {"meeting_id": meeting_id, "title": s.meta["title"], "reason": reason,
                         "error": None}
            log.info("recording %s stopped (%s) -> meeting %s", s.id, reason, meeting_id)
        threading.Thread(target=self._finish, args=(s.dir, meeting_id, s.meta),
                         daemon=True).start()
        return meeting_id

    def _finish(self, rec_dir: Path, meeting_id: str, meta: dict) -> None:
        try:
            self.sink.deliver(rec_dir, meeting_id, meta)
        except Exception as exc:   # the folder stays: it can be sent again
            log.exception("could not save recording %s", rec_dir.name)
            try:
                self.sink.fail(meeting_id, f"Saving the recording failed: {exc}")
            except Exception:
                pass
            with self.lock:
                if self.last and self.last["meeting_id"] == meeting_id:
                    self.last["error"] = f"{exc}. The audio is kept: send it again from the banner."
        finally:
            with self.lock:
                self.session, self.state = None, "idle"

    def discard(self) -> None:
        with self.lock:
            s = self.session
            if s is None or self.state != "recording":
                raise RuntimeError("not recording")
            s.halt()
            shutil.rmtree(s.dir, ignore_errors=True)
            self.session, self.state = None, "idle"
            log.info("recording %s discarded", s.id)

    def _monitor(self, s: Session) -> None:
        """Follow device changes; stop when the call ends or after MAX_S."""
        _com_init()
        call_seen = time.monotonic()
        while not s.stopped.wait(MONITOR_S):
            try:
                live_ids = {w.device.id for w in s.writers if w.is_alive()}
                for o in self.backend.outputs():
                    if o.id not in live_ids and s.failures.get(o.id, 0) < MAX_FAILURES:
                        s.start_writer(o)                 # e.g. headphones plugged in
                mic = self.backend.default_mic()
                mics = s.live("mic")
                if (mic and (not mics or mics[0].device.id != mic.id)
                        and s.failures.get(mic.id, 0) < MAX_FAILURES):
                    for w in mics:
                        w.halt.set()                     # Windows' default mic changed
                    s.start_writer(mic)
            except Exception:
                log.exception("device check failed")
            app = s.meta["call_app"]
            reason = None
            if app:
                if app in self.active_calls():
                    call_seen = time.monotonic()
                elif time.monotonic() - call_seen > CALL_END_GRACE_S:
                    reason = f"The {app} call ended"
            if s.elapsed() > MAX_S:
                reason = "Reached the 4-hour limit"
            if reason:
                try:
                    if self.session is s and self.state == "recording":
                        self.stop(reason)
                except RuntimeError:
                    pass
                return

    # --- recordings interrupted by a crash or reboot ----------------------------------------

    def interrupted(self) -> list[dict]:
        if not self.root.exists():
            return []
        active = self.session.id if self.session else None
        out = []
        for meta_path in sorted(self.root.glob("*/meta.json")):
            if meta_path.parent.name == active:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tracks = _tracks(meta_path.parent, meta)
            out.append({"id": meta["id"], "title": meta["title"],
                        "started_at": meta["started_at"],
                        "duration_s": round(rec.total_length(tracks) / SR, 1),
                        "meeting_id": meta.get("meeting_id")})
        return out

    def recover(self, rec_id: str) -> str:
        with self.lock:
            d = self._interrupted_dir(rec_id)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            meeting_id = meta.get("meeting_id")
            if not meeting_id or not self.sink.exists(meeting_id):   # none yet, or deleted
                meeting_id = self.sink.create(meta)
            meta.update(meeting_id=meeting_id,
                        stop_reason=meta.get("stop_reason") or "Recovered after an interruption")
            (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self.sink.deliver(d, meeting_id, meta)
        return meeting_id

    def drop(self, rec_id: str) -> None:
        with self.lock:
            shutil.rmtree(self._interrupted_dir(rec_id))

    def _interrupted_dir(self, rec_id: str) -> Path:
        if rec_id not in {r["id"] for r in self.interrupted()}:
            raise KeyError(rec_id)
        return self.root / rec_id


def _tracks(rec_dir: Path, meta: dict) -> list[rec.Track]:
    return [rec.Track(t["kind"], rec_dir / t["file"], t["offset"]) for t in meta["tracks"]]


def mix_recording(rec_dir: Path, src: Path, levels: Path) -> dict:
    """Mix a recording folder into one 16 kHz file plus its levels (no database)."""
    meta = json.loads((rec_dir / "meta.json").read_text(encoding="utf-8"))
    tracks = _tracks(rec_dir, meta)
    if rec.total_length(tracks) < SR:
        raise ValueError("the recording is empty (no audio was captured)")
    info = rec.mix(tracks, src, levels)
    log.info("recording %s mixed: %.0f s, echo %s", meta["id"], info["duration_s"], info["echo"])
    return info


def finalize(rec_dir: Path, meeting_id: str) -> dict:
    """Deliver a recording folder to this machine's database (used by scripts and tests)."""
    from services.recorder_local import LocalSink

    meta = json.loads((rec_dir / "meta.json").read_text(encoding="utf-8"))
    return LocalSink().deliver(rec_dir, meeting_id, meta)
