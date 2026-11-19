import { useMemo, type MouseEvent } from "react";
import { clock, seqOf, speakerIndex, withNames, type Item, type Utterance } from "../api";

const MARK: Partial<Record<Item["type"], { label: string; shape: string }>> = {
  decision: { label: "Decision", shape: "rotate-45 rounded-[1px]" },
  action: { label: "Action item", shape: "rounded-full" },
  qa: { label: "Question", shape: "rounded-[1px]" },
};

function itemText(it: Item): string {
  if (it.type === "action") return it.task;
  if (it.type === "metric") return `${it.name}: ${it.value}${it.unit ? ` ${it.unit}` : ""}`;
  if (it.type === "qa") return it.question;
  return it.text;
}

interface Props {
  transcript: Utterance[];
  duration: number;
  names: Map<string, string>;
  items: Item[];
  time: number;
  onSeek: (seconds: number, cite?: string[]) => void;
}

/** Who spoke when, across the whole meeting, with markers where notes were made. Click to seek. */
export default function Ribbon({ transcript, duration, names, items, time, onSeek }: Props) {
  const total = duration || (transcript.at(-1)?.end_ms ?? 0) / 1000 || 1;
  const startOf = useMemo(() => new Map(transcript.map((u) => [u.seq, u.start_ms / 1000])), [transcript]);
  const markers = items
    .filter((it) => MARK[it.type] && it.evidence.utterance_ids.length)
    .map((it) => ({ it, t: startOf.get(seqOf(it.evidence.utterance_ids[0])) ?? 0 }));

  const seekFromClick = (e: MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    onSeek(((e.clientX - r.left) / r.width) * total);
  };
  const pct = (s: number) => `${Math.min(100, Math.max(0, (s / total) * 100))}%`;

  return (
    <div>
      <div className="relative h-5" aria-label="Notes along the meeting">
        {markers.map(({ it, t }) => (
          <button key={it.id} type="button"
            onClick={() => onSeek(t, it.evidence.utterance_ids)}
            title={`${MARK[it.type]!.label} at ${clock(t)}: ${withNames(itemText(it), names)}`}
            aria-label={`${MARK[it.type]!.label} at ${clock(t)}: ${withNames(itemText(it), names)}`}
            className={`absolute top-1.5 size-2.5 -translate-x-1/2 ${MARK[it.type]!.shape}
              ${it.type === "decision" ? "bg-blue-ink" : it.type === "action" ? "bg-ink" : "bg-faint"}
              hover:scale-150 focus-visible:scale-150`}
            style={{ left: pct(t) }} />
        ))}
      </div>
      <div onClick={seekFromClick} aria-hidden
        className="relative h-8 cursor-pointer overflow-hidden rounded-lg bg-raised">
        {transcript.map((u) => (
          <span key={u.seq} className="absolute inset-y-0"
            style={{
              left: pct(u.start_ms / 1000),
              width: `max(1px, ${((u.end_ms - u.start_ms) / 1000 / total) * 100}%)`,
              background: `var(--spk-${speakerIndex(names, u.speaker_label)})`,
            }} />
        ))}
        <span className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_10px_white]" style={{ left: pct(time) }} />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] tabular-nums text-faint" aria-hidden>
        <span>0:00</span>
        <span className="flex gap-4">
          <span className="flex items-center gap-1.5"><span className="size-2 rotate-45 bg-blue-ink" />Decisions</span>
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-ink" />Action items</span>
          <span className="flex items-center gap-1.5"><span className="size-2 bg-faint" />Questions</span>
        </span>
        <span>{clock(total)}</span>
      </div>
    </div>
  );
}
