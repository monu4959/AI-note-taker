import { useEffect, useState } from "react";
import { Mic, Upload } from "lucide-react";
import MeetingList from "./MeetingList";
import RecordForm from "./RecordForm";
import UploadForm from "./UploadForm";

/** Landing view: what the app does, and the upload. On phones the meeting list follows. */
/** True below the lg breakpoint, where the sidebar list is hidden. */
function useSmallScreen() {
  const query = "(max-width: 1023.98px)";
  const [small, setSmall] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const on = () => setSmall(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return small;
}

export default function Home({ mode }: { mode: "record" | "upload" }) {
  const small = useSmallScreen();   // mount the second list only when it is visible
  const tab = (key: "record" | "upload", href: string, icon: React.ReactNode, text: string) => (
    <a href={href} role="tab" aria-selected={mode === key}
      className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors
        ${mode === key ? "bg-raised text-ink" : "text-muted hover:text-ink"}`}>
      {icon} {text}
    </a>
  );
  return (
    <div className="relative min-h-full">
      <div className="glow glow-in pointer-events-none absolute inset-x-0 top-0 h-[560px]" aria-hidden />
      <div className="relative mx-auto max-w-3xl px-4 pt-16 pb-16 sm:px-8 lg:pt-24">
        <h1 className="text-center text-4xl leading-[1.05] font-semibold tracking-[-0.035em] text-balance sm:text-6xl">
          Meeting notes that show their work
        </h1>
        <p className="mx-auto mt-5 max-w-xl text-center text-base text-muted text-pretty sm:text-lg">
          Record your Zoom or Teams call, or upload a recording. Get the transcript, who said
          what, and notes where every decision, action and number plays the moment it was said.
        </p>
        <div role="tablist" aria-label="Add a meeting"
          className="mx-auto mt-10 flex w-fit gap-1 rounded-xl border border-line bg-panel/80 p-1 backdrop-blur">
          {tab("record", "#/record", <Mic size={15} aria-hidden />, "Record a call")}
          {tab("upload", "#/upload", <Upload size={15} aria-hidden />, "Upload a file")}
        </div>
        <div className="mt-4" role="tabpanel">
          {mode === "record" ? <RecordForm />
            : <UploadForm onUploaded={(id) => { window.location.hash = `#/m/${id}`; }} />}
        </div>
      </div>
      {small && (
        <section aria-label="Meetings" className="border-t border-line bg-panel">
          <MeetingList />
        </section>
      )}
    </div>
  );
}
