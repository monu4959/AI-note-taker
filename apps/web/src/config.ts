import { useEffect, useState } from "react";
import { api, modelLabel, type Config } from "./api";

/** The server's config, fetched once for the whole app and shared by every component.
 *  `null` means not loaded (yet) or unavailable: callers then make no provider claims and
 *  offer no extra models. A failed request is retried the next time a component mounts. */
let loaded: Config | null = null;
let pending: Promise<Config | null> | null = null;

function load(): Promise<Config | null> {
  pending ??= api.config().then(
    (c) => (loaded = c),
    () => { pending = null; return null; },
  );
  return pending;
}

export function useConfig(): Config | null {
  const [config, setConfig] = useState<Config | null>(loaded);
  useEffect(() => {
    if (loaded) return;
    let live = true;
    load().then((c) => live && c && setConfig(c));
    return () => { live = false; };
  }, []);
  return config;
}

/** "Notes: Qwen 2.5 7B on this computer" / "Notes: Claude (Opus 5.5)"; null if unknown. */
export function notesWhere(c: Config | null): string | null {
  if (!c) return null;
  return c.notes_llm === "local"
    ? `Notes: ${modelLabel(c.extractor)} on this computer`
    : `Notes: Claude (${modelLabel(c.extractor)})`;
}
