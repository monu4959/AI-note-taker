export type MeetingStatus = "queued" | "processing" | "ready" | "failed";

export interface MeetingRow {
  id: string;
  title: string;
  status: MeetingStatus;
  duration_s: number | null;
  created_at: string;
  source_name: string | null;
  /** "live" = recorded on this computer with the desktop recorder */
  mode: "live" | "upload";
  stage: string | null;
  progress: number | null;
}

export interface Meeting extends Omit<MeetingRow, "stage" | "progress"> {
  attendees: string[];
  num_speakers: number | null;
  error: string | null;
  /** who made the recording (named from the mic track) */
  owner_name: string | null;
}

export interface Job {
  status: "queued" | "running" | "done" | "failed";
  stage: string | null;
  progress: number;
  error: string | null;
  /** set for a notes-only job that writes notes with another model */
  params?: { extractor?: string };
}

/** One set of notes for a meeting, by one model (the newest run per model). */
export interface InsightRun {
  id: number;
  model: string;
  created_at: string;
  cost_usd: number | null;
  items: number;
  status: Record<string, number>;
}

const cap = (s: string) => s[0].toUpperCase() + s.slice(1);

/** "claude-sonnet-5-5" -> "Sonnet 5.5"; local Ollama tags: "qwen2.5:7b" -> "Qwen 2.5 7B" */
export function modelLabel(model: string): string {
  const c = model.replace(/^claude-/, "").match(/^([a-z]+)-(\d+)-(\d+)$/);
  if (c) return `${cap(c[1])} ${c[2]}.${c[3]}`;
  const o = model.match(/^([a-z]+)([\d.]*):(\d+(?:\.\d+)?)b$/i);
  if (o) return `${cap(o[1])}${o[2] ? ` ${o[2]}` : ""} ${o[3]}B`;
  return model;
}

/** Claude models run in the cloud; anything else is a local model on this computer. */
export const isLocalModel = (model: string) => !model.startsWith("claude-");

/** What the server says about writing notes (GET /api/config; NOTES_LLM in .env). */
export interface Config {
  notes_llm: "local" | "claude";
  /** model ids that write and check the notes, e.g. "qwen2.5:7b" or "claude-opus-5-5" */
  extractor: string;
  verifier: string;
  /** models a meeting's notes can be (re)written with: request key -> model id */
  note_models: Record<string, string>;
}

export interface Speaker {
  cluster_label: string;
  display_name: string | null;
  /** how the name was set: by you, your mic (recordings), a remembered voice, or the transcript */
  method: "manual" | "mic" | "voiceprint" | "llm_cue" | null;
  confirmed: boolean;
  confidence: number | null;
  evidence:
    | { score: number; runner_up: number; threshold: number }
    | { kinds: string[]; utterance_ids: string[] }
    | { share: number; next_share: number | null }
    | null;
  person_id: number | null;
  sample: { start_ms: number; end_ms: number } | null;
  speech_ms: number;
}

export interface Person {
  id: number;
  display_name: string;
  voiceprints: number;
  meetings: number;
  remembered_since: string | null;
}

export interface Utterance {
  seq: number;
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  text: string;
  asr_confidence: number | null;
}

export interface MeetingDetail {
  meeting: Meeting;
  job: Job | null;
  speakers: Speaker[];
  transcript: Utterance[];
}

export interface Verification {
  status: "verified" | "needs_review" | "rejected";
  support: "supported" | "partial" | "unsupported" | null;
  reason: string | null;
  checks: { check: string; ok: boolean; hard: boolean; detail: string }[];
}

interface ItemBase {
  id: string;
  topic_id: string | null;
  evidence: { utterance_ids: string[] };
  verification: Verification;
}
export type Item =
  | (ItemBase & { type: "decision" | "key_point" | "open_question"; text: string })
  | (ItemBase & { type: "action"; task: string; owner: string | null; due: string | null })
  | (ItemBase & { type: "metric"; name: string; value: string; unit: string | null; context: string | null })
  | (ItemBase & {
      type: "qa"; question: string; asked_by: string | null; answer: string | null;
      answered_by: string | null; status: "answered" | "partial" | "unanswered";
    });

export interface Topic { id: string; title: string; start_id: string; end_id: string }

/** An answer to a question about one meeting (POST /api/meetings/{id}/ask). */
export interface Answer {
  found: boolean;
  parts: { text: string; utterance_ids: string[] }[];
  model: string;
  cost_usd: number;
}

export interface Insights {
  summary: string;
  topics: Topic[];
  items: Item[];
  meta: { extractor: string; prompt_version: number; status: Record<string, number> };
}

// ---- desktop recorder (Step 2) -----------------------------------------------------------

export interface RecorderStatus {
  state: "idle" | "recording" | "finishing";
  /** call apps using the microphone right now, e.g. ["Zoom"] */
  calls: string[];
  last: { meeting_id: string; title: string; reason: string; error: string | null } | null;
  /** recordings cut short by a crash or reboot, still on disk */
  interrupted: { id: string; title: string; started_at: string; duration_s: number }[];
  devices?: { mic: string | null; outputs: string[]; error?: string } | null;
  recording: null | {
    id: string;
    title: string;
    started_at: string;
    elapsed_s: number;
    muted: boolean;
    call_app: string | null;
    owner_name: string | null;
    levels: { mic: number; others: number };
    devices: { mic: string | null; outputs: string[] };
    warnings: string[];
  };
}

export interface RecordingStart { title: string; attendees: string; your_name: string; consent: boolean }

function post<T>(url: string, body?: unknown): Promise<T> {
  return fetch(url, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => (r.status === 204 ? (undefined as T) : json<T>(r)));
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  recorder: {
    start: (body: RecordingStart) => post<RecorderStatus>("/api/recorder/start", body),
    stop: () => post<{ meeting_id: string }>("/api/recorder/stop"),
    mute: (muted: boolean) => post<void>("/api/recorder/mute", { muted }),
    discard: () => post<void>("/api/recorder/discard"),
    recover: (id: string) => post<{ meeting_id: string }>(`/api/recorder/interrupted/${id}`),
    drop: (id: string) => fetch(`/api/recorder/interrupted/${id}`, { method: "DELETE" }).then((r) => {
      if (!r.ok) throw new Error(`${r.status}: could not delete the recording`);
    }),
  },
  meetings: () => fetch("/api/meetings").then(json<MeetingRow[]>),
  meeting: (id: string) => fetch(`/api/meetings/${id}`).then(json<MeetingDetail>),
  insights: (id: string, runId?: number) =>
    fetch(`/api/meetings/${id}/insights${runId ? `?run_id=${runId}` : ""}`).then(json<Insights>),
  insightRuns: (id: string) => fetch(`/api/meetings/${id}/insight-runs`).then(json<InsightRun[]>),
  config: () => fetch("/api/config").then(json<Config>).then((c) => {
    if ((c.notes_llm !== "local" && c.notes_llm !== "claude") || typeof c.note_models !== "object" || !c.note_models) {
      throw new Error("unexpected /api/config response");
    }
    return c;
  }),
  /** `model` is a key of Config.note_models */
  writeNotes: (id: string, model: string) =>
    fetch(`/api/meetings/${id}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    }).then(json<Job>),
  ask: (id: string, question: string, history: { question: string; answer: string }[]) =>
    post<Answer>(`/api/meetings/${id}/ask`, { question, history }),
  audioUrl: (id: string) => `/api/meetings/${id}/audio`,
  nameSpeaker: (id: string, label: string, display_name: string | null, remember_voice = false) =>
    fetch(`/api/meetings/${id}/speakers/${label}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name, remember_voice }),
    }).then(json<{ cluster_label: string; display_name: string | null; remembered: boolean }>),
  people: () => fetch("/api/people").then(json<Person[]>),
  forget: (personId: number) => fetch(`/api/people/${personId}`, { method: "DELETE" }).then((r) => {
    if (!r.ok) throw new Error(`${r.status}: could not forget this person`);
  }),
  remove: (id: string) => fetch(`/api/meetings/${id}`, { method: "DELETE" }).then((r) => {
    if (!r.ok) throw new Error(`${r.status}: delete failed`);
  }),
  /** XHR instead of fetch so the form can show upload progress. */
  upload: (form: FormData, onProgress: (fraction: number) => void) =>
    new Promise<{ id: string }>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/meetings/upload");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        if (xhr.status === 201) return resolve(JSON.parse(xhr.responseText));
        let detail = xhr.statusText;
        try { detail = JSON.parse(xhr.responseText).detail ?? detail; } catch { /* not json */ }
        reject(new Error(`${xhr.status}: ${detail}`));
      };
      xhr.onerror = () => reject(new Error("Upload failed: the notetaker server isn't running. Open the \"notetaker API\" window, or run scripts\start.ps1, then try again."));
      xhr.send(form);
    }),
};

// ---- presentation helpers --------------------------------------------------------------

export const uttId = (seq: number) => `u${String(seq).padStart(4, "0")}`;
export const seqOf = (id: string) => Number(id.replace(/^u/, ""));

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export function duration(seconds: number | null): string {
  if (!seconds) return "";
  const m = Math.round(seconds / 60);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

/** Speaker names: the user's name if set, else "Speaker N" in order of first appearance. */
export function speakerNames(speakers: Speaker[], transcript: Utterance[]): Map<string, string> {
  const order: string[] = [];
  for (const u of transcript) if (!order.includes(u.speaker_label)) order.push(u.speaker_label);
  for (const s of speakers) if (!order.includes(s.cluster_label)) order.push(s.cluster_label);
  const names = new Map<string, string>();
  order.forEach((label, i) => {
    const s = speakers.find((x) => x.cluster_label === label);
    names.set(label, s?.display_name || `Speaker ${i + 1}`);
  });
  return names;
}

export function speakerIndex(names: Map<string, string>, label: string): number {
  return Math.max(0, [...names.keys()].indexOf(label)) % 8;
}

/** Replace diarization labels (SPEAKER_02) inside note text with display names. */
export function withNames(text: string, names: Map<string, string>): string {
  return text.replace(/SPEAKER_\d{2}/g, (label) => names.get(label) ?? label);
}
