import { useEffect, useState, type FormEvent } from "react";
import { Headphones, Mic, Speaker } from "lucide-react";
import { api, modelLabel } from "../api";
import { useConfig } from "../config";
import { notifyCallsEnabled, setNotifyCalls, useRecorder } from "../recorder";

const NAME_KEY = "notetaker.yourName";
const CONSENT_KEY = "notetaker.teamConsent";   // ticked once per person, kept on this computer
const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
const read = (k: string) => { try { return localStorage.getItem(k) ?? ""; } catch { return ""; } };
const write = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } };

/** Start recording a call on this computer: your mic plus everything the computer plays. */
export default function RecordForm() {
  const { status, connected } = useRecorder();
  const config = useConfig();
  const [yourName, setYourName] = useState(() => read(NAME_KEY));
  const [consent, setConsent] = useState(() => read(CONSENT_KEY) === "1");
  const [notify, setNotify] = useState(notifyCallsEnabled);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { write(NAME_KEY, yourName.trim()); }, [yourName]);
  useEffect(() => { write(CONSENT_KEY, consent ? "1" : "0"); }, [consent]);

  const call = status?.calls[0];
  const devices = status?.devices;
  const recording = status?.state === "recording";

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      await api.recorder.start({
        title: String(f.get("title") ?? ""), attendees: String(f.get("attendees") ?? ""),
        your_name: yourName, consent,
      });
    } catch (err) {
      setError((err as Error).message.replace(/^\d+: /, ""));
    } finally {
      setBusy(false);
    }
  };

  const field = "mt-1.5 w-full rounded-xl border border-line bg-bg px-3 py-2 text-ink outline-none placeholder:text-faint focus:border-blue-ink";
  const label = "block text-sm font-medium text-muted";

  // The server's page opened from another computer: recording would capture the server's
  // sound, so it is refused there. Point to the recorder app instead.
  if (!LOCAL) {
    return (
      <div className="rounded-3xl border border-line-strong bg-panel/80 p-6 backdrop-blur">
        <p className="font-medium">Record from the recorder app on your computer</p>
        <p className="mt-2 text-sm text-muted">
          Recording has to run on the computer that is in the call. Start the Notetaker recorder
          (see the setup guide) and open <a className="text-blue-ink underline" href="http://localhost:8765">http://localhost:8765</a>.
          Everything else (meetings, notes, uploads) works from here too.
        </p>
      </div>
    );
  }

  if (recording) {
    return (
      <div className="rounded-3xl border border-line-strong bg-panel/80 p-6 text-center backdrop-blur">
        <p className="font-medium">Recording “{status!.recording!.title}”</p>
        <p className="mt-1 text-sm text-muted">Use the bar at the top to pause your mic or stop. You can leave this page.</p>
      </div>
    );
  }

  return (
    <form onSubmit={submit} aria-label="Record a call"
      className="rounded-3xl border border-line-strong bg-panel/80 p-4 shadow-[0_30px_80px_-30px_rgb(46_75_255/0.45)] backdrop-blur sm:p-6">
      <div className="rounded-2xl border border-line bg-bg/60 p-4 text-sm">
        <p className="flex items-start gap-2.5">
          <Mic size={16} className="mt-0.5 shrink-0 text-blue-ink" aria-hidden />
          <span><span className="text-ink">Your microphone</span>
            <span className="block text-faint">{devices?.mic ?? (connected ? "No microphone found" : "Connecting…")}</span></span>
        </p>
        <p className="mt-2 flex items-start gap-2.5">
          <Speaker size={16} className="mt-0.5 shrink-0 text-blue-ink" aria-hidden />
          <span><span className="text-ink">Everything this computer plays</span>, so everyone else on the call
            <span className="block text-faint">{devices?.outputs.length ? devices.outputs.join(", ") : connected ? "No speakers found" : "Connecting…"}</span></span>
        </p>
        <p className="mt-2 flex items-start gap-2.5 text-muted">
          <Headphones size={16} className="mt-0.5 shrink-0" aria-hidden />
          Works with or without headphones. Keep the call's sound on, and pause music: whatever
          the computer plays ends up in the recording.
        </p>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className={label}>Title
          <input name="title" className={field} placeholder={`${call ? `${call} call` : "Meeting"}, ${new Date().toLocaleDateString(undefined, { month: "short", day: "numeric" })}`} />
        </label>
        <label className={label}>Your name
          <input value={yourName} onChange={(e) => setYourName(e.target.value)} className={field}
            placeholder="So your lines are labelled" autoComplete="name" />
        </label>
        <label className={`${label} sm:col-span-2`}>Attendees
          <input name="attendees" className={field} placeholder="Priya, Raj, Maya" />
          <span className="mt-1.5 block text-xs text-faint">Optional. Helps spell names and match remembered voices.</span>
        </label>
      </div>

      <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-xl border border-line p-3 text-sm hover:bg-hover">
        <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)}
          className="mt-0.5 size-4 accent-[var(--blue)]" />
        <span>
          <span className="font-medium text-ink">My team knows our calls are recorded</span>
          <span className="mt-0.5 block text-faint">
            Asked once, remembered on this computer. If someone outside the team joins (a client,
            a candidate), tell them too: in Massachusetts and several other states everyone on a
            call has to agree.
          </span>
        </span>
      </label>

      {error && <p role="alert" className="mt-4 text-sm text-danger">{error}</p>}
      <button type="submit" disabled={busy || !consent || !connected}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-blue px-4 py-3 font-semibold text-white shadow-[0_0_30px_rgb(46_75_255/0.45)] transition hover:brightness-110 disabled:opacity-50 disabled:shadow-none">
        <span className="size-2.5 rounded-full bg-white" aria-hidden /> {busy ? "Starting…" : "Start recording"}
      </button>
      <p className="mt-3 text-center text-xs text-faint">
        {call ? `It stops by itself when the ${call} call ends. ` : "Stop it from the bar at the top when the meeting ends. "}
        Audio goes only to your team's notetaker server
        {config?.notes_llm === "claude" ? "; the transcript text is sent to Claude to write the notes."
          : config?.notes_llm === "local" ? `, and so does the transcript: ${modelLabel(config.extractor)} writes the notes there.`
          : "."}
      </p>

      <label className="mt-4 flex cursor-pointer items-center justify-center gap-2 text-xs text-muted">
        <input type="checkbox" checked={notify} className="size-3.5 accent-[var(--blue)]"
          onChange={async (e) => setNotify(await setNotifyCalls(e.target.checked))} />
        Notify me when a Zoom or Teams call starts (while this tab is open)
      </label>
    </form>
  );
}
