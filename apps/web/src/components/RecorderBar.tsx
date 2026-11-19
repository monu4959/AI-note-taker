import { useEffect, useState } from "react";
import { Mic, MicOff, PhoneCall, RotateCcw, Square, Trash2, TriangleAlert } from "lucide-react";
import { api, clock, duration } from "../api";
import { useRecorder } from "../recorder";

/** dBFS -> 0..100 % of a level meter (-60 dB is empty, -10 dB is full). */
const meter = (db: number) => Math.max(0, Math.min(100, ((db + 60) / 50) * 100));

function Meter({ label, db, off }: { label: string; db: number; off?: boolean }) {
  return (
    <span className="flex items-center gap-2 text-xs text-muted">
      <span className="w-11 shrink-0">{label}</span>
      <span className="relative h-1.5 w-16 overflow-hidden rounded-full bg-line-strong sm:w-24" aria-hidden>
        <span className={`absolute inset-y-0 left-0 rounded-full transition-[width] duration-150 motion-reduce:transition-none ${off ? "bg-faint" : "bg-blue-ink"}`}
          style={{ width: `${off ? 0 : meter(db)}%` }} />
      </span>
    </span>
  );
}

const strip = "flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-panel px-4 py-2.5 sm:px-8";
const ghost = "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-muted hover:bg-hover hover:text-ink disabled:opacity-50";

/** Top of every page: the recording in progress, or what needs your attention about one. */
export default function RecorderBar({ onRecordPage }: { onRecordPage: boolean }) {
  const { status } = useRecorder();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [dismissedCall, setDismissedCall] = useState<string | null>(null);
  const [seenStop, setSeenStop] = useState<string | null>(null);

  // a new call (after the last one ended) can prompt again
  const callKey = status?.calls.join(",") || null;
  useEffect(() => { if (!callKey) setDismissedCall(null); }, [callKey]);
  useEffect(() => { if (status?.state !== "recording") setConfirmDiscard(false); }, [status?.state]);

  if (!status) return null;
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const errorLine = error && <p role="alert" className="w-full text-sm text-danger">{error}</p>;

  const r = status.recording;
  if (status.state === "recording" && r) {
    return (
      <div className={strip} role="region" aria-label="Recording in progress">
        <span className="flex min-w-0 items-center gap-2.5">
          <span className="relative flex size-2.5 shrink-0" aria-hidden>
            <span className="absolute inset-0 animate-ping rounded-full bg-danger/70 motion-reduce:animate-none" />
            <span className="relative size-2.5 rounded-full bg-danger" />
          </span>
          <span className="truncate font-medium">{r.title}</span>
          <span className="tabular-nums text-muted" aria-label={`Recording for ${clock(r.elapsed_s)}`}>{clock(r.elapsed_s)}</span>
        </span>
        <span className="flex items-center gap-4">
          <Meter label="You" db={r.levels.mic} off={r.muted} />
          <Meter label="Others" db={r.levels.others} />
        </span>
        <span className="ml-auto flex items-center gap-1">
          {confirmDiscard ? (
            <>
              <span className="mr-1 text-sm text-muted">Delete this recording?</span>
              <button type="button" disabled={busy} onClick={() => run(api.recorder.discard)}
                className="rounded-lg bg-danger/15 px-2.5 py-1.5 text-sm font-medium text-danger hover:bg-danger/25">Delete</button>
              <button type="button" onClick={() => setConfirmDiscard(false)} className={ghost}>Keep recording</button>
            </>
          ) : (
            <>
              <button type="button" aria-pressed={r.muted} disabled={busy}
                onClick={() => run(() => api.recorder.mute(!r.muted))} className={ghost}
                title={r.muted ? "Your voice is not being recorded" : "Stop recording your voice; the others are still recorded"}>
                {r.muted ? <MicOff size={15} className="text-danger" aria-hidden /> : <Mic size={15} aria-hidden />}
                {r.muted ? "Resume my mic" : "Pause my mic"}
              </button>
              <button type="button" onClick={() => setConfirmDiscard(true)} className={ghost} aria-label="Delete this recording">
                <Trash2 size={15} aria-hidden />
              </button>
              <button type="button" disabled={busy}
                onClick={() => run(async () => { const { meeting_id } = await api.recorder.stop(); if (meeting_id) window.location.hash = `#/m/${meeting_id}`; })}
                className="flex items-center gap-1.5 rounded-lg bg-blue px-3 py-1.5 text-sm font-semibold text-white shadow-[0_0_20px_rgb(46_75_255/0.45)] hover:brightness-110 disabled:opacity-60">
                <Square size={13} fill="currentColor" aria-hidden /> Stop and take notes
              </button>
            </>
          )}
        </span>
        {r.warnings.map((w) => (
          <p key={w} className="flex w-full items-center gap-2 text-sm text-amber">
            <TriangleAlert size={14} aria-hidden /> {w}
          </p>
        ))}
        {errorLine}
      </div>
    );
  }

  if (status.state === "finishing") {
    return <div className={strip} role="status"><span className="text-sm text-muted">Saving the recording…</span></div>;
  }

  // idle: the most recent stop, if it was automatic or failed
  const last = status.last;
  const lastKey = last ? `${last.meeting_id ?? ""}:${last.title}:${last.error ?? ""}` : null;
  if (last && lastKey !== seenStop && (last.error || last.reason !== "Stopped by you")) {
    return (
      <div className={strip} role="status">
        <span className="text-sm">
          {last.error ? <span className="text-danger">Saving “{last.title}” failed: {last.error}</span>
            : <>Recording stopped: {last.reason}. Notes are on the way.</>}
        </span>
        <span className="ml-auto flex gap-1">
          {last.meeting_id && !last.error && (
            <a href={`#/m/${last.meeting_id}`} onClick={() => setSeenStop(lastKey)}
              className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-blue-ink hover:bg-hover">Open the meeting</a>
          )}
          <button type="button" onClick={() => setSeenStop(lastKey)} className={ghost}>Dismiss</button>
        </span>
      </div>
    );
  }

  const cut = status.interrupted[0];
  if (cut) {
    return (
      <div className={strip} role="status">
        <span className="flex items-center gap-2 text-sm">
          <RotateCcw size={15} className="text-blue-ink" aria-hidden />
          “{cut.title}” hasn't been turned into notes yet ({duration(cut.duration_s) || `${Math.round(cut.duration_s)} s`} recorded; cut short or not sent). The audio is still here.
        </span>
        <span className="ml-auto flex gap-1">
          <button type="button" disabled={busy}
            onClick={() => run(async () => { const { meeting_id } = await api.recorder.recover(cut.id); window.location.hash = `#/m/${meeting_id}`; })}
            className="rounded-lg px-2.5 py-1.5 text-sm font-medium text-blue-ink hover:bg-hover">Take notes from it</button>
          <button type="button" disabled={busy} onClick={() => run(() => api.recorder.drop(cut.id))} className={ghost}>Delete</button>
        </span>
        {errorLine}
      </div>
    );
  }

  const call = status.calls[0];
  if (call && callKey !== dismissedCall && !onRecordPage) {
    return (
      <div className={strip} role="status">
        <span className="flex items-center gap-2 text-sm">
          <PhoneCall size={15} className="text-blue-ink" aria-hidden />
          {call} is using your microphone. Record this call?
        </span>
        <span className="ml-auto flex gap-1">
          <a href="#/record" className="rounded-lg bg-blue px-3 py-1.5 text-sm font-semibold text-white hover:brightness-110">Record it</a>
          <button type="button" onClick={() => setDismissedCall(callKey)} className={ghost}>Not now</button>
        </span>
      </div>
    );
  }
  return null;
}
