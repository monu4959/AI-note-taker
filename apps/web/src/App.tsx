import { useEffect, useState } from "react";
import { AudioLines, ListVideo, Plus, Users } from "lucide-react";
import Home from "./components/Home";
import MeetingList from "./components/MeetingList";
import MeetingView from "./components/MeetingView";
import People from "./components/People";
import RecorderBar from "./components/RecorderBar";
import { RecorderProvider, useRecorder } from "./recorder";

function useHashRoute(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const onChange = () => setHash(window.location.hash);
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return hash;
}

function RailLink({ href, label, active, children }: {
  href: string; label: string; active: boolean; children: React.ReactNode;
}) {
  return (
    <a href={href} aria-label={label} title={label} aria-current={active ? "page" : undefined}
      className={`grid size-10 place-items-center rounded-xl transition-colors
        ${active ? "bg-raised text-ink" : "text-faint hover:bg-hover hover:text-ink"}`}>
      {children}
    </a>
  );
}

export default function App() {
  return <RecorderProvider><Shell /></RecorderProvider>;
}

function Shell() {
  const hash = useHashRoute();
  const meetingId = hash.match(/^#\/m\/([\w-]+)/)?.[1];
  const page = meetingId ? "meeting" : hash.startsWith("#/people") ? "people" : "home";
  const recording = useRecorder().status?.state === "recording";

  return (
    <div className="flex h-dvh flex-col overflow-hidden lg:flex-row">
      {/* Icon rail (a top bar on small screens) */}
      <nav aria-label="Main"
        className="flex shrink-0 items-center gap-2 border-b border-line bg-panel px-3 py-2 lg:w-16 lg:flex-col lg:border-r lg:border-b-0 lg:px-0 lg:py-4">
        <a href="#/" aria-label="Notetaker home" title="Notetaker"
          className="mr-2 grid size-10 place-items-center rounded-xl bg-blue text-white lg:mr-0 lg:mb-4">
          <AudioLines size={20} strokeWidth={2.25} />
        </a>
        <RailLink href="#/" label="Meetings" active={page !== "people"}><ListVideo size={19} /></RailLink>
        <RailLink href="#/people" label="Remembered voices" active={page === "people"}><Users size={19} /></RailLink>
        <a href="#/record" aria-label={recording ? "Recording in progress" : "Record or upload a meeting"}
          title={recording ? "Recording in progress" : "Record or upload a meeting"}
          className="relative ml-auto grid size-10 place-items-center rounded-full bg-blue text-white shadow-[0_0_24px_rgb(46_75_255/0.55)] hover:brightness-110 lg:mt-auto lg:ml-0">
          <Plus size={20} strokeWidth={2.5} />
          {recording && <span className="absolute -top-0.5 -right-0.5 size-3 rounded-full border-2 border-panel bg-danger" aria-hidden />}
        </a>
      </nav>

      {/* Meeting list pane (large screens; on phones it is part of the home page) */}
      <aside className="hidden w-[300px] shrink-0 flex-col border-r border-line bg-panel lg:flex">
        <MeetingList activeId={meetingId} />
      </aside>

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden" id="main">
        <RecorderBar onRecordPage={page === "home" && !hash.startsWith("#/upload")} />
        <div className="min-h-0 flex-1 overflow-y-auto">
          {page === "meeting" && <MeetingView id={meetingId!} key={meetingId} />}
          {page === "people" && <People />}
          {page === "home" && <Home mode={hash.startsWith("#/upload") ? "upload" : "record"} />}
        </div>
      </main>
    </div>
  );
}
