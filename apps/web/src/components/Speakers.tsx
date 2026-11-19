import { useEffect, useMemo, useState, type FormEvent } from "react";
import { AudioWaveform, Check, Fingerprint, MessageSquareQuote, Mic, Pencil, UserRound, Volume2 } from "lucide-react";
import { api, clock, seqOf, speakerIndex, type Person, type Speaker, type Utterance } from "../api";

interface Props {
  meetingId: string;
  speakers: Speaker[];
  transcript: Utterance[];
  names: Map<string, string>;
  onChanged: () => void;
  onPlay: (seconds: number, cite?: string[], until?: number | null) => void;
}

function minutes(ms: number) {
  const m = ms / 60000;
  return m < 1 ? `${Math.round(ms / 1000)} s` : `${m.toFixed(m < 10 ? 1 : 0)} min`;
}

/** Why a speaker has the name it has, in words. */
function Source({ s }: { s: Speaker }) {
  if (s.method === "manual") {
    return (
      <span className="flex items-center gap-1.5 text-muted">
        <Check size={14} className="text-blue-ink" aria-hidden />
        Named by you{s.person_id ? ", voice remembered" : ""}
      </span>
    );
  }
  if (s.method === "mic" && s.evidence && "share" in s.evidence) {
    return (
      <span className="flex items-center gap-1.5 text-muted">
        <Mic size={14} className="text-blue-ink" aria-hidden />
        Your microphone: {Math.round(s.evidence.share * 100)}% of this voice was only on your mic
      </span>
    );
  }
  if (s.method === "voiceprint" && s.evidence && "score" in s.evidence) {
    return (
      <span className="flex items-center gap-1.5 text-muted">
        <Fingerprint size={14} className="text-blue-ink" aria-hidden />
        Recognised by voice, similarity {s.evidence.score.toFixed(2)} (needs {s.evidence.threshold.toFixed(2)})
      </span>
    );
  }
  if (s.method === "llm_cue" && s.evidence && "utterance_ids" in s.evidence) {
    return (
      <span className="flex items-center gap-1.5 text-muted">
        <MessageSquareQuote size={14} className="text-amber" aria-hidden />
        From the transcript, please confirm
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-faint">
      <UserRound size={14} aria-hidden /> Not identified
    </span>
  );
}

/** The transcript line a proposed name came from, playable. */
function Quote({ s, transcript, onPlay }: { s: Speaker; transcript: Utterance[]; onPlay: Props["onPlay"] }) {
  if (s.method !== "llm_cue" || !s.evidence || !("utterance_ids" in s.evidence)) return null;
  const ids = s.evidence.utterance_ids;
  const first = transcript.find((u) => u.seq === seqOf(ids[0]));
  if (!first) return null;
  return (
    <button type="button" onClick={() => onPlay(first.start_ms / 1000, ids)}
      className="mt-3 block w-full max-w-prose rounded-lg border-l-2 border-amber/60 bg-amber-bg px-3 py-2 text-left text-sm text-ink/85 hover:text-ink sm:ml-13 sm:w-auto">
      “{first.text.length > 200 ? `${first.text.slice(0, 200)}…` : first.text}”
      <span className="ml-2 text-xs tabular-nums text-faint">{clock(first.start_ms / 1000)}</span>
    </button>
  );
}

/** Step 3: who is who. Automatic names are proposals until you confirm them; remembering a
 *  voice is opt-in and can be undone under Remembered voices. */
export default function Speakers({ meetingId, speakers, transcript, names, onChanged, onPlay }: Props) {
  const [editing, setEditing] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [people, setPeople] = useState<Person[]>([]);
  useEffect(() => { api.people().then(setPeople).catch(() => setPeople([])); }, []);

  const byLabel = useMemo(() => new Map(speakers.map((s) => [s.cluster_label, s])), [speakers]);
  const ordered = [...names.keys()].map((l) => byLabel.get(l)).filter((s): s is Speaker => !!s);
  const toConfirm = ordered.filter((s) => s.display_name && !s.confirmed).length;

  const save = async (e: FormEvent<HTMLFormElement>, label: string) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const value = String(f.get("name") ?? "").trim();
    setBusy(true);
    try {
      await api.nameSpeaker(meetingId, label, value || null, f.get("remember") === "on");
      setEditing(null);
      setError(null);
      onChanged();
      api.people().then(setPeople).catch(() => {});
    } catch (err) {
      setError((err as Error).message.replace(/^\d+: /, ""));
    } finally {
      setBusy(false);
    }
  };

  const confirm = async (s: Speaker) => {
    setBusy(true);
    try {
      await api.nameSpeaker(meetingId, s.cluster_label, s.display_name, false);
      onChanged();
    } catch (err) {
      setError((err as Error).message.replace(/^\d+: /, ""));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section aria-labelledby="speakers-h">
      <h2 id="speakers-h" className="sr-only">Speakers</h2>
      <p className="max-w-prose text-muted">
        Names come from voices you asked the app to remember, or from the transcript when someone
        is introduced or called by name. Unclear speakers stay unnamed rather than guessed.
        {toConfirm > 0 && <span className="text-ink"> {toConfirm} {toConfirm === 1 ? "name needs" : "names need"} your confirmation.</span>}
      </p>
      {error && <p role="alert" className="mt-3 text-sm text-danger">{error}</p>}
      <datalist id="known-people">
        {people.map((p) => <option key={p.id} value={p.display_name} />)}
      </datalist>

      <ul className="mt-5 divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel">
        {ordered.map((s) => {
          const label = s.cluster_label;
          const name = names.get(label)!;
          const hue = `var(--spk-${speakerIndex(names, label)})`;
          return (
            <li key={label} className="p-4 sm:px-5">
              <div className="flex flex-wrap items-start gap-x-4 gap-y-3">
                <span className="mt-1 grid size-9 shrink-0 place-items-center rounded-full text-sm font-semibold text-bg"
                  style={{ background: hue }} aria-hidden>
                  {s.display_name ? s.display_name.slice(0, 1).toUpperCase() : <AudioWaveform size={16} />}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="flex flex-wrap items-baseline gap-x-2">
                    <span className={`text-base font-semibold ${s.display_name ? "text-ink" : "text-muted"}`}>{name}</span>
                    <span className="text-xs text-faint">{minutes(s.speech_ms)} of speech</span>
                  </p>
                  <div className="mt-1 text-sm"><Source s={s} /></div>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  {s.sample && (
                    <button type="button" onClick={() => onPlay(s.sample!.start_ms / 1000, [], s.sample!.end_ms / 1000)}
                      className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-sm text-muted hover:border-line-strong hover:text-ink"
                      aria-label={`Play a sample of ${name}`}>
                      <Volume2 size={14} aria-hidden /> Sample
                    </button>
                  )}
                  {s.display_name && !s.confirmed && (
                    <button type="button" disabled={busy} onClick={() => confirm(s)}
                      className="flex items-center gap-1.5 rounded-lg bg-blue px-2.5 py-1.5 text-sm font-medium text-white hover:brightness-110 disabled:opacity-60">
                      <Check size={14} aria-hidden /> Confirm
                    </button>
                  )}
                  <button type="button" onClick={() => setEditing(editing === label ? null : label)}
                    aria-expanded={editing === label}
                    className="flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-sm text-muted hover:border-line-strong hover:text-ink">
                    <Pencil size={14} aria-hidden /> {s.display_name ? "Rename" : "Name"}
                  </button>
                </div>
              </div>

              <Quote s={s} transcript={transcript} onPlay={onPlay} />
              {editing === label && (
                <form onSubmit={(e) => save(e, label)} className="mt-4 rounded-xl bg-raised p-4 sm:ml-13">
                  <label className="block text-sm font-medium text-muted">
                    Name
                    <input name="name" list="known-people" autoFocus defaultValue={s.display_name ?? ""}
                      placeholder="Leave empty to make this speaker anonymous again"
                      className="mt-1.5 w-full rounded-xl border border-line bg-bg px-3 py-2 text-ink outline-none placeholder:text-faint focus:border-blue-ink" />
                  </label>
                  <label className="mt-3 flex items-start gap-2.5 text-sm text-muted">
                    <input type="checkbox" name="remember" defaultChecked={!!s.person_id}
                      className="mt-0.5 size-4 accent-[var(--blue)]" />
                    <span>
                      <span className="text-ink">Remember this voice</span> so future meetings recognise
                      this person. Stores a voiceprint on this machine; only do this with their
                      agreement. You can forget it any time.
                    </span>
                  </label>
                  <div className="mt-4 flex gap-2">
                    <button type="submit" disabled={busy}
                      className="rounded-lg bg-blue px-3.5 py-1.5 text-sm font-medium text-white hover:brightness-110 disabled:opacity-60">
                      Save name
                    </button>
                    <button type="button" onClick={() => setEditing(null)}
                      className="rounded-lg px-3.5 py-1.5 text-sm text-muted hover:bg-hover hover:text-ink">
                      Cancel
                    </button>
                  </div>
                </form>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
