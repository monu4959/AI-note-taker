import { useEffect, useRef, useState } from "react";
import { clock, speakerIndex, type Utterance } from "../api";

interface Props {
  transcript: Utterance[];
  names: Map<string, string>;
  time: number;
  cited: { seqs: number[]; nonce: number };
  onSeek: (seconds: number) => void;
}

export default function Transcript({ transcript, names, time, cited, onSeek }: Props) {
  const [follow, setFollow] = useState(true);
  const list = useRef<HTMLOListElement>(null);
  const active = transcript.find((u) => time * 1000 >= u.start_ms && time * 1000 < u.end_ms + 600)?.seq;

  // A citation click scrolls to the first cited line (flash is applied via the nonce key).
  useEffect(() => {
    if (!cited.nonce || !cited.seqs.length) return;
    list.current?.querySelector(`[data-seq="${cited.seqs[0]}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [cited]);

  // While playing, keep the current line in view unless the reader turned that off.
  useEffect(() => {
    if (!follow || active === undefined) return;
    const el = list.current?.querySelector<HTMLElement>(`[data-seq="${active}"]`);
    const box = list.current?.getBoundingClientRect();
    const r = el?.getBoundingClientRect();
    if (el && box && r && (r.top < box.top || r.bottom > box.bottom)) el.scrollIntoView({ block: "nearest" });
  }, [active, follow]);

  return (
    <section aria-labelledby="transcript-h" className="rounded-2xl border border-line bg-panel p-3 xl:sticky xl:top-40 xl:self-start">
      <div className="flex items-baseline justify-between px-2 pt-1">
        <h2 id="transcript-h" className="text-sm font-semibold">Transcript</h2>
        <label className="flex items-center gap-2 text-xs text-muted">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} className="accent-[var(--blue)]" />
          Follow playback
        </label>
      </div>
      <ol ref={list} className="mt-2 max-h-[70vh] overflow-y-auto pr-1 xl:max-h-[calc(100dvh-13rem)]">
        {transcript.map((u) => {
          const isCited = cited.seqs.includes(u.seq);
          return (
            <li key={isCited ? `${u.seq}-${cited.nonce}` : u.seq} data-seq={u.seq}
              className={`rounded-lg px-2 py-1.5 ${isCited ? "cite-flash" : ""} ${u.seq === active ? "bg-raised" : ""}`}>
              <div className="flex items-baseline gap-2 text-sm">
                <span className="size-2 shrink-0 rounded-full" style={{ background: `var(--spk-${speakerIndex(names, u.speaker_label)})` }} />
                <span className="font-semibold">{names.get(u.speaker_label)}</span>
                <button type="button" onClick={() => onSeek(u.start_ms / 1000)}
                  className="text-xs tabular-nums text-faint hover:text-blue-ink" aria-label={`Play from ${clock(u.start_ms / 1000)}`}>
                  {clock(u.start_ms / 1000)}
                </button>
              </div>
              <p className="mt-0.5 leading-relaxed text-ink/85">{u.text}</p>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
