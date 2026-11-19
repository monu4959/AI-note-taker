import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { RecorderStatus } from "./api";

/** The desktop recorder's live status, shared by the whole app over ONE server-sent-events
 *  connection (levels 4x a second while recording; call detection otherwise). */
const Ctx = createContext<{ status: RecorderStatus | null; connected: boolean }>({ status: null, connected: false });

export const useRecorder = () => useContext(Ctx);

const NOTIFY_KEY = "notetaker.notifyCalls";
export const notifyCallsEnabled = () => {
  try { return localStorage.getItem(NOTIFY_KEY) === "1" && Notification.permission === "granted"; } catch { return false; }
};
export async function setNotifyCalls(on: boolean): Promise<boolean> {
  if (on && "Notification" in window && Notification.permission !== "granted") {
    if ((await Notification.requestPermission()) !== "granted") on = false;
  }
  try { localStorage.setItem(NOTIFY_KEY, on ? "1" : "0"); } catch { /* private mode */ }
  return on;
}

function notify(title: string, body: string, href?: string) {
  if (!document.hidden || !notifyCallsEnabled()) return;
  const n = new Notification(title, { body, icon: "/favicon.svg" });
  n.onclick = () => { window.focus(); if (href) window.location.hash = href; n.close(); };
}

export function RecorderProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<RecorderStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const prev = useRef<RecorderStatus | null>(null);

  useEffect(() => {
    // recording runs on this computer only: the server's page opened from elsewhere has none
    if (!["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname)) return;
    const es = new EventSource("/api/recorder/events");
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);   // EventSource reconnects by itself
    es.onmessage = (e) => {
      const s = JSON.parse(e.data) as RecorderStatus;
      const p = prev.current;
      if (p) {
        const call = s.calls.find((c) => !p.calls.includes(c));
        if (call && s.state === "idle") notify(`${call} call started`, "Open the notetaker to record it.", "#/");
        if (p.state === "recording" && s.state !== "recording" && s.last && s.last.reason !== "Stopped by you") {
          notify("Recording stopped", `${s.last.reason}. Notes are on the way.`, `#/m/${s.last.meeting_id}`);
        }
      }
      prev.current = s;
      setStatus(s);
    };
    return () => es.close();
  }, []);

  return <Ctx.Provider value={{ status, connected }}>{children}</Ctx.Provider>;
}
