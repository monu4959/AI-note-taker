import { useCallback, useEffect, useState } from "react";
import { Fingerprint, Trash2 } from "lucide-react";
import { api, type Person } from "../api";

/** Remembered voices. Voiceprints are biometric data: opt-in per person, deletable here. */
export default function People() {
  const [people, setPeople] = useState<Person[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.people().then((p) => { setPeople(p); setError(null); }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => { load(); }, [load]);

  const forget = async (p: Person) => {
    if (!window.confirm(`Forget ${p.display_name}? This deletes their stored voiceprints. Names you typed in meetings stay; names that came from their voice are removed.`)) return;
    try { await api.forget(p.id); load(); } catch (e) { setError((e as Error).message); }
  };

  return (
    <div className="mx-auto max-w-3xl px-4 pt-8 pb-16 sm:px-8">
      <h1 className="text-3xl font-semibold tracking-[-0.03em]">Remembered voices</h1>
      <p className="mt-3 max-w-prose text-muted">
        When you name a speaker and tick “Remember this voice”, the app stores a voiceprint on
        this machine and recognises that person in later meetings. Only remember people who
        agreed to it. Forgetting someone deletes every stored voiceprint.
      </p>
      {error && <p role="alert" className="mt-4 text-sm text-danger">{error}</p>}
      {people && people.length === 0 && (
        <div className="mt-8 rounded-2xl border border-dashed border-line-strong p-8 text-center">
          <Fingerprint size={28} className="mx-auto text-faint" aria-hidden />
          <p className="mt-3 font-medium">No voices remembered yet</p>
          <p className="mt-1 text-sm text-muted">
            Open a meeting, go to Speakers, and name someone with “Remember this voice” ticked.
          </p>
        </div>
      )}
      {people && people.length > 0 && (
        <ul className="mt-8 divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel">
          {people.map((p) => (
            <li key={p.id} className="flex items-center gap-4 p-4 sm:px-5">
              <span className="grid size-9 shrink-0 place-items-center rounded-full bg-raised text-sm font-semibold">
                {p.display_name.slice(0, 1).toUpperCase()}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-semibold">{p.display_name}</p>
                <p className="text-sm text-muted">
                  {p.voiceprints} {p.voiceprints === 1 ? "voiceprint" : "voiceprints"}, named in {p.meetings}{" "}
                  {p.meetings === 1 ? "meeting" : "meetings"}
                  {p.remembered_since && `, remembered since ${new Date(p.remembered_since).toLocaleDateString()}`}
                </p>
              </div>
              <button type="button" onClick={() => forget(p)}
                className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm text-faint hover:bg-hover hover:text-danger">
                <Trash2 size={15} aria-hidden /> Forget
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
