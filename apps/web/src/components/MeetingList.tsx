import { useCallback, useEffect, useState } from "react";
import { Cloud, Cpu, Mic, Search } from "lucide-react";
import { api, duration, type MeetingRow } from "../api";
import { notesWhere, useConfig } from "../config";

export const STAGE_LABEL: Record<string, string> = {
  prepare: "Preparing audio",
  transcribe: "Transcribing",
  diarize: "Separating speakers",
  assign: "Matching words to speakers",
  names: "Finding names",
  insights: "Writing and checking notes",
};

function Status({ m }: { m: MeetingRow }) {
  if (m.status === "ready") return <span className="text-faint">{duration(m.duration_s)}</span>;
  if (m.status === "failed") return <span className="font-medium text-danger">Processing failed</span>;
  const pct = Math.round((m.progress ?? 0) * 100);
  return (
    <span className="flex items-center gap-2 text-muted">
      <span className="h-1 w-14 overflow-hidden rounded-full bg-line-strong" aria-hidden>
        <span className="block h-full bg-blue transition-[width]" style={{ width: `${pct}%` }} />
      </span>
      {m.status === "queued" ? "Waiting" : STAGE_LABEL[m.stage ?? ""] ?? "Processing"}
    </span>
  );
}

/** All meetings, newest first, with a title filter. Polls faster while something processes. */
export default function MeetingList({ activeId }: { activeId?: string }) {
  const [rows, setRows] = useState<MeetingRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const config = useConfig();
  const where = notesWhere(config);   // where notes are written: a quiet footer line

  const load = useCallback(() => {
    api.meetings().then((r) => { setRows(r); setError(null); }).catch((e) => setError(String(e.message)));
  }, []);

  // Poll: every 2 s while something is processing, else every 10 s. The interval depends only
  // on `busy` (a boolean), not on `rows`, so a fresh response never triggers another fetch.
  const busy = rows?.some((r) => r.status === "queued" || r.status === "processing") ?? false;
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const t = setInterval(load, busy ? 2000 : 10000);
    return () => clearInterval(t);
  }, [load, busy]);

  const shown = rows?.filter((m) => m.title.toLowerCase().includes(q.trim().toLowerCase()));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="p-3">
        <label className="flex items-center gap-2 rounded-xl border border-line bg-bg px-3 py-2 text-muted focus-within:border-line-strong">
          <Search size={15} aria-hidden />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search meetings"
            aria-label="Search meetings" className="w-full bg-transparent text-ink outline-none placeholder:text-faint" />
        </label>
      </div>
      <p className="px-4 pb-2 text-xs text-faint">
        {rows ? `All meetings (${rows.length})` : "Loading meetings…"}
      </p>
      {error && (
        <p className="mx-4 rounded-lg bg-raised p-3 text-sm text-muted">
          The server isn't responding. Start it with <code className="text-ink">scripts\start.ps1</code>.
        </p>
      )}
      {rows && rows.length === 0 && (
        <p className="mx-4 text-sm text-muted">No meetings yet. Upload a recording to get started.</p>
      )}
      {shown && rows && rows.length > 0 && shown.length === 0 && (
        <p className="mx-4 text-sm text-muted">No meeting titles match "{q}".</p>
      )}
      <ul className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-4">
        {shown?.map((m) => (
          <li key={m.id}>
            <a href={`#/m/${m.id}`} aria-current={m.id === activeId ? "page" : undefined}
              className={`block rounded-xl px-3 py-2.5 transition-colors
                ${m.id === activeId ? "bg-raised" : "hover:bg-hover"}`}>
              <span className="block text-[11px] text-faint">
                {new Date(m.created_at).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}
              </span>
              <span className="mt-0.5 flex items-center gap-1.5 font-medium text-ink">
                {m.mode === "live" && <Mic size={13} className="shrink-0 text-faint" aria-label="Recorded on this computer" />}
                <span className="truncate">{m.title}</span>
              </span>
              <span className="mt-0.5 block text-xs"><Status m={m} /></span>
            </a>
          </li>
        ))}
      </ul>
      {where && (
        <p className="flex items-center gap-1.5 border-t border-line px-4 py-3 text-xs text-faint" data-testid="notes-where">
          {config!.notes_llm === "local" ? <Cpu size={13} aria-hidden /> : <Cloud size={13} aria-hidden />}
          {where}
        </p>
      )}
    </div>
  );
}
