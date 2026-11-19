import { Check } from "lucide-react";
import { modelLabel, type Config, type Job, type Meeting } from "../api";
import { useConfig } from "../config";

/** Time per hour of audio: the local notes model is slower than Claude. */
function estimate(c: Config | null): string {
  if (!c) return "This takes a few minutes per hour of audio on this machine.";
  return c.notes_llm === "local"
    ? "About 10–15 minutes per hour of audio on this machine."
    : "About 4–7 minutes per hour of audio on this machine.";
}

/** Who writes the notes, shown quietly next to that step. */
function notesBy(c: Config | null): string | null {
  if (!c) return null;
  return c.notes_llm === "local"
    ? `${modelLabel(c.extractor)} on this computer`
    : `Claude ${modelLabel(c.extractor)}, checked by ${modelLabel(c.verifier)}`;
}

// The pipeline's stages, in order (this really is a sequence, so it is numbered).
const STEPS: [string, string][] = [
  ["prepare", "Prepare the audio"],
  ["transcribe", "Transcribe speech"],
  ["diarize", "Separate the speakers"],
  ["assign", "Match words to speakers"],
  ["names", "Look for names: remembered voices and introductions"],
  ["insights", "Write the notes and check each one against the transcript"],
];

export default function Processing({ meeting, job }: { meeting: Meeting; job: Job | null }) {
  const config = useConfig();
  const by = notesBy(config);
  if (meeting.status === "failed") {
    return (
      <div className="mx-4 mt-8 max-w-prose rounded-2xl border border-danger/30 bg-panel p-6 sm:mx-8">
        <h2 className="text-lg font-semibold">Processing failed</h2>
        <p className="mt-2 text-muted">{meeting.error ?? job?.error ?? "The worker reported an error."}</p>
        <p className="mt-2 text-muted">
          Check the worker window for details{meeting.mode === "live" ? "." : <>, then <a href="#/upload" className="text-blue-ink underline">upload the recording again</a>.</>}
        </p>
      </div>
    );
  }
  if (meeting.mode === "live" && !job) {
    return (
      <div className="mx-4 mt-8 max-w-xl rounded-2xl border border-line bg-panel p-6 sm:mx-8">
        <h2 className="text-lg font-semibold">Saving the recording</h2>
        <p className="mt-2 text-sm text-muted">
          Lining up your mic with the call audio and removing speaker echo. Notes start right after.
        </p>
      </div>
    );
  }
  const current = STEPS.findIndex(([key]) => key === job?.stage);
  const pct = Math.round((job?.progress ?? 0) * 100);
  return (
    <div className="mx-4 mt-8 max-w-xl rounded-2xl border border-line bg-panel p-6 sm:mx-8">
      <h2 className="text-lg font-semibold">
        {meeting.status === "queued" ? "Waiting for the worker to start" : `Taking notes, ${pct}% done`}
      </h2>
      <div className="mt-3 h-1 overflow-hidden rounded-full bg-raised" aria-hidden>
        <div className="h-full bg-blue shadow-[0_0_12px_var(--blue)] transition-[width] duration-700" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-3 text-sm text-faint">
        {estimate(config)} You can leave this page; it keeps going.
      </p>
      <ol className="mt-6 space-y-3">
        {STEPS.map(([key, label], i) => {
          const state = i < current ? "done" : i === current ? (pct >= 100 ? "done" : "now") : "later";
          return (
            <li key={key} className="flex items-center gap-3">
              <span className={`grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold
                ${state === "done" ? "bg-blue text-white" : state === "now" ? "border-2 border-blue-ink text-blue-ink" : "border border-line-strong text-faint"}`}>
                {state === "done" ? <Check size={13} strokeWidth={3} /> : i + 1}
              </span>
              <span className={state === "later" ? "text-faint" : "text-ink"}>
                {label}
                {key === "insights" && by && <span className="mt-0.5 block text-xs text-faint">{by}</span>}
              </span>
            </li>
          );
        })}
      </ol>
      {job?.status === "queued" && (
        <p className="mt-6 text-sm text-muted">
          Nothing is processing yet. Make sure the worker window is open (<code className="text-ink">scripts\start.ps1</code>).
        </p>
      )}
    </div>
  );
}
