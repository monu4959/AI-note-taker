import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, clock, modelLabel, seqOf, withNames, type Answer, type Topic, type Utterance } from "../api";

export interface Turn { question: string; answer?: Answer; error?: string }

interface Props {
  meetingId: string;
  transcript: Utterance[];
  names: Map<string, string>;
  topics: Topic[];
  chat: Turn[];
  setChat: (update: (chat: Turn[]) => Turn[]) => void;
  onCite: (seconds: number, cite: string[]) => void;
}

const plain = (a: Answer) => a.parts.map((p) => p.text).join(" ");

/** Ask about the meeting. Answers come only from this meeting and cite the lines they use. */
export default function Ask({ meetingId, transcript, names, topics, chat, setChat, onCite }: Props) {
  const [draft, setDraft] = useState("");
  const busy = chat.some((t) => !t.answer && !t.error);
  const startOf = useMemo(() => new Map(transcript.map((u) => [u.seq, u.start_ms / 1000])), [transcript]);
  const end = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  useEffect(() => { if (chat.length) end.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }); }, [chat]);

  const suggestions = [
    "What was decided, and why?",
    "Which questions were left unanswered?",
    "Who is doing what next?",
    ...topics.slice(1, 3).map((t) => `What was said about ${t.title.charAt(0).toLowerCase()}${t.title.slice(1)}?`),
  ];

  const send = async (question: string) => {
    const q = question.trim();
    if (!q || busy) return;
    const history = chat.filter((t) => t.answer).map((t) => ({ question: t.question, answer: plain(t.answer!) }));
    setDraft("");
    setChat((c) => [...c, { question: q }]);
    try {
      const answer = await api.ask(meetingId, q, history);
      setChat((c) => c.map((t, i) => (i === c.length - 1 ? { ...t, answer } : t)));
    } catch (e) {
      setChat((c) => c.map((t, i) => (i === c.length - 1 ? { ...t, error: (e as Error).message.replace(/^\d+: /, "") } : t)));
    }
    input.current?.focus();
  };
  const submit = (e: FormEvent) => { e.preventDefault(); send(draft); };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(draft); }
  };

  return (
    <section aria-label="Ask about this meeting" className="max-w-[68ch]">
      {chat.length === 0 && (
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Ask about this meeting</h2>
          <p className="mt-1 text-sm text-muted">
            Answers come only from what was said, with the moments they're based on. Ask about a number,
            a person, or anything the notes leave out.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <button key={s} type="button" onClick={() => send(s)}
                className="rounded-lg border border-line bg-panel px-3 py-1.5 text-left text-sm text-ink/90 hover:border-blue-ink hover:text-ink">
                {withNames(s, names)}
              </button>
            ))}
          </div>
        </div>
      )}

      <ol className="space-y-7" aria-live="polite">
        {chat.map((t, i) => (
          <li key={i}>
            <p className="ml-auto w-fit max-w-[90%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-ink">{t.question}</p>
            <div className="mt-3">
              {t.error ? (
                <p role="alert" className="text-sm text-amber">{t.error}</p>
              ) : !t.answer ? (
                <p className="text-sm text-muted">
                  <span className="mr-2 inline-block size-2 animate-pulse rounded-full bg-blue-ink align-middle" />
                  Reading the meeting…
                </p>
              ) : (
                <>
                  <p className={`leading-[1.75] ${t.answer.found ? "text-ink/90" : "text-muted"}`}>
                    {t.answer.parts.map((p, j) => (
                      <span key={j}>
                        {withNames(p.text, names)}
                        {p.utterance_ids.slice(0, 3).map((id) => {
                          const at = startOf.get(seqOf(id));
                          return at === undefined ? null : (
                            <button key={id} type="button" onClick={() => onCite(at, p.utterance_ids)}
                              aria-label={`Play from ${clock(at)}`}
                              className="ml-1.5 rounded-md border border-line bg-raised px-1.5 align-baseline text-xs tabular-nums text-muted hover:border-blue-ink hover:text-blue-ink">
                              {clock(at)}
                            </button>
                          );
                        })}{" "}
                      </span>
                    ))}
                  </p>
                  <p className="mt-1.5 text-xs text-faint">
                    {t.answer.found ? "From the transcript" : "Not found in this meeting"}, {modelLabel(t.answer.model)}
                    {t.answer.cost_usd ? `, $${t.answer.cost_usd.toFixed(3)}` : ", free"}
                  </p>
                </>
              )}
            </div>
          </li>
        ))}
      </ol>
      <div ref={end} />

      <form onSubmit={submit} className="sticky bottom-4 mt-6 flex items-end gap-2 rounded-2xl border border-line bg-panel p-2 shadow-[0_8px_30px_rgba(0,0,0,.35)]">
        <label htmlFor="ask-input" className="sr-only">Your question</label>
        <textarea id="ask-input" ref={input} rows={1} value={draft} onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey} maxLength={1000}
          placeholder={chat.length ? "Ask a follow-up…" : "e.g. What did we say the budget was?"}
          className="max-h-40 min-h-10 flex-1 resize-none bg-transparent px-2.5 py-2 text-ink placeholder:text-faint focus:outline-none" />
        <button type="submit" disabled={busy || !draft.trim()}
          className="rounded-xl bg-blue px-4 py-2 text-sm font-semibold text-white disabled:opacity-40">
          Ask
        </button>
      </form>
      {chat.length > 0 && !busy && (
        <button type="button" onClick={() => setChat(() => [])} className="mt-3 text-sm text-faint hover:text-muted">
          Clear conversation
        </button>
      )}
    </section>
  );
}
