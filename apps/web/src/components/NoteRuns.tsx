import { Columns2, LoaderCircle, Sparkles } from "lucide-react";
import { isLocalModel, modelLabel, type InsightRun, type Job } from "../api";
import { useConfig } from "../config";

interface Props {
  runs: InsightRun[];
  selected: number | null;
  compare: boolean;
  job: Job | null;
  error: string | null;
  onSelect: (runId: number) => void;
  onCompare: (on: boolean) => void;
  /** a key of the server's note_models, e.g. "local" or "sonnet" */
  onWrite: (modelKey: string) => void;
}

function cost(r: InsightRun): string {
  if (r.cost_usd === 0 || (r.cost_usd == null && isLocalModel(r.model))) return ", free, written on this computer";
  return r.cost_usd != null ? `, $${r.cost_usd.toFixed(2)}` : "";
}

export function runSummary(r: InsightRun): string {
  const check = r.status?.needs_review ?? 0;
  return `${r.items} notes, ${check} to check${cost(r)}`;
}

function writeHint(model: string): string {
  if (isLocalModel(model)) return "free, on this computer, reuses the transcript";
  return model.includes("sonnet") ? "about half the cost, reuses the transcript" : "reuses the transcript";
}

/** Which model's notes are shown, and writing another set with a model the server offers.
 *  Only the server's note_models are offered, so with local notes Claude never is; while
 *  the config is unknown nothing is offered. */
export default function NoteRuns({ runs, selected, compare, job, error, onSelect, onCompare, onWrite }: Props) {
  const config = useConfig();
  const writing = job?.params?.extractor && (job.status === "queued" || job.status === "running");
  const missing = Object.entries(config?.note_models ?? {})
    .filter(([, model]) => !runs.some((r) => r.model === model));

  return (
    <div className="mb-6 flex flex-wrap items-center gap-2">
      {!compare && runs.map((r) => (
        <button key={r.id} type="button" aria-pressed={r.id === selected} onClick={() => onSelect(r.id)}
          className={`rounded-xl border px-3 py-1.5 text-left text-sm transition-colors
            ${r.id === selected ? "border-blue-ink/60 bg-blue/15 text-ink" : "border-line text-muted hover:border-line-strong hover:text-ink"}`}>
          <span className="font-medium">Notes by {modelLabel(r.model)}</span>
          <span className="ml-2 text-xs text-faint">{runSummary(r)}</span>
        </button>
      ))}
      {runs.length > 1 && (
        <button type="button" aria-pressed={compare} onClick={() => onCompare(!compare)}
          className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-sm transition-colors
            ${compare ? "border-blue-ink/60 bg-blue/15 text-ink" : "border-line text-muted hover:border-line-strong hover:text-ink"}`}>
          <Columns2 size={15} aria-hidden /> {compare ? "Show one set" : "Compare side by side"}
        </button>
      )}
      {writing ? (
        <span className="flex items-center gap-2 rounded-xl border border-line px-3 py-1.5 text-sm text-muted" role="status">
          <LoaderCircle size={15} className="animate-spin text-blue-ink motion-reduce:animate-none" aria-hidden />
          {modelLabel(job!.params!.extractor!)} is writing notes
          {job!.status === "running" ? `, ${Math.round(job!.progress * 100)}%` : ", waiting for the worker"}
        </span>
      ) : missing.map(([key, model]) => (
        <button key={key} type="button" onClick={() => onWrite(key)}
          className="flex items-center gap-1.5 rounded-xl border border-dashed border-line-strong px-3 py-1.5 text-sm text-muted hover:border-blue-ink hover:text-ink">
          <Sparkles size={15} className="text-blue-ink" aria-hidden />
          {runs.length ? "Also write" : "Write"} notes with {modelLabel(model)}
          <span className="text-xs text-faint">{writeHint(model)}</span>
        </button>
      ))}
      {error && <p role="alert" className="w-full text-sm text-danger">{error}</p>}
    </div>
  );
}
