import { useMemo, useState, type ReactNode } from "react";
import { clock, seqOf, withNames, type Insights, type Item, type Topic, type Utterance } from "../api";

interface Props {
  insights: Insights;
  transcript: Utterance[];
  names: Map<string, string>;
  onCite: (seconds: number, cite: string[]) => void;
}

type Of<T extends Item["type"]> = Item & { type: T };

/** ✓ for a checked note (unless quiet), "Check this" with the reason when the evidence is thin. */
function Mark({ it, quiet = false }: { it: Item; quiet?: boolean }) {
  const [open, setOpen] = useState(false);
  const v = it.verification;
  if (v.status === "verified") {
    return quiet ? null : (
      <span className="ml-1.5 text-sm text-blue-ink" title="Checked against the cited lines">✓<span className="sr-only">verified</span></span>
    );
  }
  const flags = v.checks.filter((c) => !c.ok).map((c) => c.detail).filter(Boolean);
  return (
    <span className="ml-1.5">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="rounded-md bg-amber-bg px-1.5 py-px text-xs font-semibold text-amber">
        Check this
      </button>
      {open && (
        <span className="mt-1 block text-sm text-muted">
          {[v.reason, ...flags].filter(Boolean).join(" ")}
        </span>
      )}
    </span>
  );
}

function Cites({ ids, startOf, onCite, max = 4 }: {
  ids: string[]; startOf: Map<number, number>; onCite: Props["onCite"]; max?: number;
}) {
  const times = ids.map((id) => ({ id, t: startOf.get(seqOf(id)) })).filter((x) => x.t !== undefined);
  return (
    <span className="ml-1.5 inline-flex flex-wrap gap-1 align-baseline">
      {times.slice(0, max).map(({ id, t }) => (
        <button key={id} type="button" onClick={() => onCite(t!, ids)}
          className="rounded-md border border-line bg-raised px-1.5 text-xs tabular-nums text-muted hover:border-blue-ink hover:text-blue-ink"
          aria-label={`Play from ${clock(t!)}`}>
          {clock(t!)}
        </button>
      ))}
    </span>
  );
}

function Section({ title, count, empty, children }: { title: string; count: number; empty: string; children: ReactNode }) {
  const id = `sec-${title.replace(/\W+/g, "-")}`;
  return (
    <section aria-labelledby={id} className="mt-12 border-t border-line pt-6">
      <h2 id={id} className="text-sm font-semibold tracking-tight">
        {title} <span className="ml-1 font-normal text-faint">{count}</span>
      </h2>
      {count === 0 ? <p className="mt-2 text-sm text-faint">{empty}</p> : children}
    </section>
  );
}

const STATUS_LABEL = { answered: null, partial: "Partly answered", unanswered: "Not answered" } as const;

export default function Notes({ insights, transcript, names, onCite }: Props) {
  const startOf = useMemo(() => new Map(transcript.map((u) => [u.seq, u.start_ms / 1000])), [transcript]);
  const n = (s: string | null) => (s ? withNames(s, names) : "");
  const cites = (it: Item, max?: number) => (
    <Cites ids={it.evidence.utterance_ids} startOf={startOf} onCite={onCite} max={max} />
  );

  // Every note belongs to a topic: its own, or the topic whose span holds its first citation
  // (single-pass notes and older runs leave topic_id empty for decisions and actions).
  const byTopic = useMemo(() => {
    const spans = insights.topics.map((t) => ({ id: t.id, a: seqOf(t.start_id), b: seqOf(t.end_id) }));
    const known = new Set(spans.map((s) => s.id));
    const map = new Map<string, Item[]>();
    for (const it of insights.items) {
      const first = it.evidence.utterance_ids[0];
      const q = first ? seqOf(first) : NaN;
      const tid = it.topic_id && known.has(it.topic_id)
        ? it.topic_id
        : spans.find((s) => q >= s.a && q <= s.b)?.id ?? "";
      map.set(tid, [...(map.get(tid) ?? []), it]);
    }
    return map;
  }, [insights]);

  const all = <T extends Item["type"]>(type: T) => insights.items.filter((i): i is Of<T> => i.type === type);
  const actions = all("action");
  const stillOpen: (Of<"open_question"> | Of<"qa">)[] = [
    ...all("qa").filter((q) => q.status === "unanswered"),
    ...all("open_question"),
  ];
  const toCheck = insights.items.filter((i) => i.verification.status === "needs_review").length;
  const loose = byTopic.get("") ?? [];
  const topics: (Topic & { loose?: boolean })[] = [
    ...insights.topics,
    ...(loose.some((i) => i.type !== "action" && i.type !== "open_question") ? [{ id: "", title: "Other notes", start_id: "", end_id: "", loose: true }] : []),
  ];

  return (
    <div className="text-[16px]">
      <p className="max-w-[68ch] text-[17px] leading-[1.7] text-ink/90">{n(insights.summary)}</p>
      <p className="mt-4 text-sm text-faint">
        {insights.items.length - toCheck} notes checked against the transcript
        {toCheck ? `, ${toCheck} marked "Check this" where the evidence is incomplete` : ""}.
      </p>

      {topics.map((t) => {
        const items = byTopic.get(t.id) ?? [];
        const pick = <T extends Item["type"]>(type: T) => items.filter((i): i is Of<T> => i.type === type);
        const story = pick("key_point");
        const decisions = pick("decision");
        const qa = pick("qa");
        const numbers = pick("metric");
        const at = t.loose ? undefined : startOf.get(seqOf(t.start_id));
        return (
          <section key={t.id || "loose"} aria-labelledby={`topic-${t.id || "loose"}`}
            className="mt-12 border-t border-line pt-6">
            <h2 id={`topic-${t.id || "loose"}`} className="flex flex-wrap items-baseline gap-x-3 text-lg font-semibold tracking-tight">
              {n(t.title)}
              {at !== undefined && (
                <button type="button" onClick={() => onCite(at, [t.start_id])}
                  className="text-sm font-normal tabular-nums text-faint hover:text-blue-ink"
                  aria-label={`Play topic from ${clock(at)}`}>
                  {clock(at)}
                </button>
              )}
            </h2>

            {story.length > 0 && (
              <p className="mt-3 max-w-[68ch] leading-[1.75] text-ink/90">
                {story.map((it) => (
                  <span key={it.id}>{n(it.text.trim().replace(/([^.!?])$/, "$1."))}<Mark it={it} quiet />{cites(it, 1)}{" "}</span>
                ))}
              </p>
            )}

            {decisions.length > 0 && (
              <div className="mt-5 max-w-[68ch]">
                <h3 className="text-sm font-semibold text-muted">Decided</h3>
                <ul className="mt-2 space-y-2 leading-relaxed text-ink/90">
                  {decisions.map((it) => (
                    <li key={it.id} className="border-l-2 border-blue-ink pl-3">{n(it.text)}<Mark it={it} />{cites(it)}</li>
                  ))}
                </ul>
              </div>
            )}

            {qa.length > 0 && (
              <div className="mt-5 max-w-[68ch]">
                <h3 className="text-sm font-semibold text-muted">Questions asked</h3>
                <ul className="mt-2 space-y-4">
                  {qa.map((it) => {
                    const label = STATUS_LABEL[it.status];
                    return (
                      <li key={it.id} className="leading-relaxed">
                        <p className="font-medium text-ink">
                          {n(it.question)}
                          {it.asked_by && <span className="ml-2 text-sm font-normal text-muted">asked by {n(it.asked_by)}</span>}
                          {label && (
                            <span className="ml-2 rounded-md border border-line px-1.5 py-px text-xs font-normal text-muted">{label}</span>
                          )}
                        </p>
                        {it.answer && (
                          <p className="mt-1 text-ink/85">
                            {it.answered_by && <span className="text-muted">{n(it.answered_by)}: </span>}
                            {n(it.answer)}
                          </p>
                        )}
                        <span className="-ml-1.5 mt-1 flex flex-wrap items-baseline"><Mark it={it} />{cites(it)}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {numbers.length > 0 && (
              <details className="mt-5 max-w-[68ch]">
                <summary className="cursor-pointer text-sm text-faint hover:text-muted">
                  Numbers mentioned ({numbers.length})
                </summary>
                <ul className="mt-2 divide-y divide-line">
                  {numbers.map((it) => (
                    <li key={it.id} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-2">
                      <span className="min-w-0 flex-1 basis-56 text-sm text-ink/90">
                        {n(it.name)}
                        {it.context && <span className="block text-muted">{n(it.context)}</span>}
                      </span>
                      <span className="text-sm font-semibold tabular-nums text-ink">
                        {it.value}{it.unit ? ` ${it.unit}` : ""}
                      </span>
                      <span className="flex flex-wrap items-baseline"><Mark it={it} />{cites(it)}</span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </section>
        );
      })}

      <Section title="Action items" count={actions.length} empty="No action items in this meeting.">
        <ul className="mt-3 max-w-[68ch] space-y-3 leading-relaxed text-ink/90">
          {actions.map((it) => (
            <li key={it.id}>
              {n(it.task)}
              <span className="ml-2 text-sm text-muted">
                {it.owner ? n(it.owner) : "No owner named"}{it.due ? `, ${it.due}` : ""}
              </span>
              <Mark it={it} />{cites(it)}
            </li>
          ))}
        </ul>
      </Section>

      <Section title="Still open" count={stillOpen.length} empty="Nothing left open.">
        <ul className="mt-3 max-w-[68ch] space-y-3 leading-relaxed text-ink/90">
          {stillOpen.map((it) => (
            <li key={it.id}>
              {n(it.type === "qa" ? it.question : it.text)}
              {it.type === "qa" && it.asked_by && <span className="ml-2 text-sm text-muted">asked by {n(it.asked_by)}</span>}
              <Mark it={it} />{cites(it)}
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}
