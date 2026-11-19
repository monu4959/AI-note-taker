import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, Trash2 } from "lucide-react";
import { api, clock, duration, modelLabel, seqOf, speakerIndex, speakerNames,
  type InsightRun, type Insights, type MeetingDetail } from "../api";
import { useConfig } from "../config";
import NoteRuns, { runSummary } from "./NoteRuns";
import Notes from "./Notes";
import Ask, { type Turn } from "./Ask";
import Processing from "./Processing";
import Ribbon from "./Ribbon";
import Speakers from "./Speakers";
import Transcript from "./Transcript";

const SPEEDS = [1, 1.25, 1.5, 2];

export default function MeetingView({ id }: { id: string }) {
  const config = useConfig();
  const [detail, setDetail] = useState<MeetingDetail | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [runs, setRuns] = useState<InsightRun[]>([]);
  const [runId, setRunId] = useState<number | null>(null);
  const [compare, setCompare] = useState(false);
  const [other, setOther] = useState<Insights | null>(null);   // the second set, in compare mode
  const [notesError, setNotesError] = useState<string | null>(null);
  const notesJobWasActive = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tab, setTab] = useState<"notes" | "ask" | "speakers">("notes");
  const [chat, setChat] = useState<Turn[]>([]);   // kept while switching tabs
  useEffect(() => { setChat([]); }, [id]);
  const [cited, setCited] = useState<{ seqs: number[]; nonce: number }>({ seqs: [], nonce: 0 });
  const audio = useRef<HTMLAudioElement>(null);
  const stopAt = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const d = await api.meeting(id);
      setDetail(d);
      if (d.meeting.status === "ready") {
        const r = await api.insightRuns(id);
        setRuns((old) => (JSON.stringify(old) === JSON.stringify(r) ? old : r));
        // a second set of notes just finished: show both side by side
        const active = !!d.job?.params?.extractor && (d.job.status === "queued" || d.job.status === "running");
        if (notesJobWasActive.current && !active && r.length > 1) setCompare(true);
        notesJobWasActive.current = active;
      }
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id]);

  useEffect(() => { load(); }, [load]);
  const status = detail?.meeting.status;
  const job = detail?.job ?? null;
  const notesBusy = !!job?.params?.extractor && (job.status === "queued" || job.status === "running");
  useEffect(() => {
    if (status !== "queued" && status !== "processing" && !notesBusy) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [status, notesBusy, load]);

  // Which run is shown: the one picked, else the server's current notes model, else Opus,
  // else the newest.
  const shownRun = runs.find((r) => r.id === runId)
    ?? runs.find((r) => r.model === config?.extractor)
    ?? runs.find((r) => r.model.includes("opus")) ?? runs[0];
  const otherRun = runs.find((r) => r !== shownRun);
  useEffect(() => {
    if (!shownRun) return;
    let live = true;
    api.insights(id, shownRun.id).then((x) => live && setInsights(x)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [id, shownRun?.id]);
  useEffect(() => {
    if (!compare || !otherRun) { setOther(null); return; }
    let live = true;
    api.insights(id, otherRun.id).then((x) => live && setOther(x)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [id, compare, otherRun?.id]);

  const writeNotes = async (model: string) => {
    setNotesError(null);
    try {
      await api.writeNotes(id, model);
      notesJobWasActive.current = true;
      await load();
    } catch (e) {
      setNotesError((e as Error).message.replace(/^\d+: /, ""));
    }
  };

  const names = useMemo(
    () => (detail ? speakerNames(detail.speakers, detail.transcript) : new Map<string, string>()),
    [detail],
  );

  /** Jump the audio to a time and point the transcript at the cited lines. */
  const seek = useCallback((seconds: number, citeIds: string[] = [], until: number | null = null) => {
    const a = audio.current;
    stopAt.current = until;
    if (a) {
      a.currentTime = seconds;
      void a.play().catch(() => { /* autoplay blocked: position is still set */ });
    }
    setTime(seconds);
    if (citeIds.length) setCited((c) => ({ seqs: citeIds.map(seqOf), nonce: c.nonce + 1 }));
  }, []);

  const toggle = () => {
    const a = audio.current;
    if (!a) return;
    stopAt.current = null;
    if (a.paused) void a.play(); else a.pause();
  };

  if (error && !detail) {
    return (
      <p className="p-8 text-muted">
        This meeting couldn't be loaded ({error}). <a href="#/" className="text-blue-ink underline">Back to meetings</a>
      </p>
    );
  }
  if (!detail) return <p className="p-8 text-muted">Loading…</p>;

  const m = detail.meeting;
  const total = m.duration_s ?? 0;
  const named = detail.speakers.filter((s) => s.display_name).length;
  const remove = async () => {
    if (!window.confirm(`Delete "${m.title}"? This removes the recording, transcript and notes.`)) return;
    await api.remove(id);
    window.location.hash = "#/";
  };

  return (
    <article className="min-h-full">
      <header className="flex flex-wrap items-start justify-between gap-4 px-4 pt-6 sm:px-8 lg:pt-8">
        <div className="min-w-0">
          <p className="text-xs text-faint">
            {new Date(m.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
            {total ? `, ${duration(total)}` : ""}
            {m.status === "ready" ? `, ${detail.speakers.length} speakers` : ""}
          </p>
          <h1 className="mt-1 text-3xl font-semibold tracking-[-0.03em] text-balance sm:text-4xl">{m.title}</h1>
        </div>
        <button onClick={remove}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-faint hover:bg-hover hover:text-danger">
          <Trash2 size={15} aria-hidden /> Delete meeting
        </button>
      </header>

      {m.status !== "ready" ? (
        <Processing meeting={m} job={detail.job} />
      ) : (
        <>
          <audio ref={audio} src={api.audioUrl(id)} preload="metadata"
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => {
              const t = e.currentTarget.currentTime;
              setTime(t);
              if (stopAt.current !== null && t >= stopAt.current) { e.currentTarget.pause(); stopAt.current = null; }
            }} />

          {/* Player: play, time, speed; the ribbon below is the scrubber */}
          <div className="sticky top-0 z-20 border-b border-line bg-bg/85 px-4 pt-4 pb-3 backdrop-blur-md sm:px-8">
            <div className="flex items-center gap-3">
              <button type="button" onClick={toggle} aria-label={playing ? "Pause" : "Play"}
                className="grid size-10 shrink-0 place-items-center rounded-full bg-blue text-white shadow-[0_0_22px_rgb(46_75_255/0.5)] hover:brightness-110">
                {playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" className="ml-0.5" />}
              </button>
              <span className="text-sm tabular-nums text-muted">
                <span className="text-ink">{clock(time)}</span> / {clock(total)}
              </span>
              <button type="button"
                onClick={() => {
                  const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
                  setSpeed(next);
                  if (audio.current) audio.current.playbackRate = next;
                }}
                aria-label={`Playback speed ${speed}x`}
                className="rounded-lg border border-line px-2 py-0.5 text-xs tabular-nums text-muted hover:border-line-strong hover:text-ink">
                {speed}×
              </button>
              <ul className="ml-auto hidden max-w-[55%] flex-wrap justify-end gap-x-3 gap-y-1 text-xs text-muted md:flex">
                {[...names.entries()].slice(0, 8).map(([label, name]) => (
                  <li key={label} className="flex items-center gap-1.5">
                    <span className="size-2 rounded-full" style={{ background: `var(--spk-${speakerIndex(names, label)})` }} />
                    {name}
                  </li>
                ))}
                {names.size > 8 && <li className="text-faint">+{names.size - 8} more</li>}
              </ul>
            </div>
            <div className="mt-3">
              <Ribbon transcript={detail.transcript} duration={total} names={names}
                items={insights?.items ?? []} time={time} onSeek={seek} />
            </div>
          </div>

          <div className={`grid gap-10 px-4 pt-6 pb-16 sm:px-8 ${compare && tab === "notes" ? "" : "xl:grid-cols-[minmax(0,1fr)_400px]"}`}>
            <div className="min-w-0">
              <div role="tablist" aria-label="Meeting views" className="inline-flex rounded-xl border border-line bg-panel p-1">
                {([["notes", "Notes"], ["ask", "Ask"], ["speakers", `Speakers  ${named}/${detail.speakers.length} named`]] as const).map(([key, label]) => (
                  <button key={key} role="tab" type="button" aria-selected={tab === key} onClick={() => setTab(key)}
                    className={`rounded-lg px-3.5 py-1.5 text-sm whitespace-pre transition-colors
                      ${tab === key ? "bg-raised font-medium text-ink" : "text-muted hover:text-ink"}`}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="mt-6">
                {tab === "notes" ? (
                  <>
                    <NoteRuns runs={runs} selected={shownRun?.id ?? null} compare={compare} job={job}
                      error={notesError} onSelect={setRunId} onCompare={setCompare} onWrite={writeNotes} />
                    {compare && shownRun && otherRun ? (
                      <div className="grid gap-8 lg:grid-cols-2">
                        {([[shownRun, insights], [otherRun, other]] as const).map(([r, x]) => (
                          <section key={r.id} aria-label={`Notes by ${modelLabel(r.model)}`}
                            className="min-w-0 rounded-2xl border border-line bg-panel/60 p-5">
                            <h2 className="text-base font-semibold">Notes by {modelLabel(r.model)}</h2>
                            <p className="mt-0.5 mb-5 text-xs text-faint">{runSummary(r)}</p>
                            {x ? <Notes insights={x} transcript={detail.transcript} names={names} onCite={seek} />
                              : <p className="text-muted">Loading notes…</p>}
                          </section>
                        ))}
                      </div>
                    ) : insights ? (
                      <Notes insights={insights} transcript={detail.transcript} names={names} onCite={seek} />
                    ) : (
                      <p className="text-muted">Loading notes…</p>
                    )}
                  </>
                ) : tab === "ask" ? (
                  <Ask meetingId={id} transcript={detail.transcript} names={names}
                    topics={insights?.topics ?? []} chat={chat} setChat={setChat} onCite={seek} />
                ) : (
                  <Speakers meetingId={id} speakers={detail.speakers} transcript={detail.transcript}
                    names={names} onChanged={load} onPlay={seek} />
                )}
              </div>
            </div>
            {!(compare && tab === "notes") && (
              <Transcript transcript={detail.transcript} names={names} time={time} cited={cited} onSeek={seek} />
            )}
          </div>
        </>
      )}
    </article>
  );
}
