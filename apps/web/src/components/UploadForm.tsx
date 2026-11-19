import { useRef, useState, type DragEvent, type FormEvent } from "react";
import { FileAudio, UploadCloud } from "lucide-react";
import { api, modelLabel } from "../api";
import { useConfig } from "../config";

const ACCEPT = ".wav,.mp3,.m4a,.mp4,.webm,.ogg,.flac,.aac,.mkv,.mov";

export default function UploadForm({ onUploaded }: { onUploaded: (id: string) => void }) {
  const config = useConfig();
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  };

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) return setError("Choose a recording first.");
    const form = new FormData(e.currentTarget);
    form.set("file", file);
    if (!form.get("num_speakers")) form.delete("num_speakers");
    setError(null);
    setProgress(0);
    try {
      const { id } = await api.upload(form, setProgress);
      onUploaded(id);
    } catch (err) {
      setError((err as Error).message);
      setProgress(null);
    }
  };

  const field = "mt-1.5 w-full rounded-xl border border-line bg-bg px-3 py-2 text-ink outline-none placeholder:text-faint focus:border-blue-ink";
  const label = "block text-sm font-medium text-muted";

  return (
    <form onSubmit={submit} aria-label="Upload a recording"
      className="rounded-3xl border border-line-strong bg-panel/80 p-4 shadow-[0_30px_80px_-30px_rgb(46_75_255/0.45)] backdrop-blur sm:p-6">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => input.current?.click()}
        className={`flex cursor-pointer flex-col items-center rounded-2xl border border-dashed px-4 py-8 text-center transition-colors
          ${dragging ? "border-blue-ink bg-blue/10" : "border-line-strong hover:bg-hover"}`}
      >
        {file ? <FileAudio size={28} className="text-blue-ink" aria-hidden /> : <UploadCloud size={28} className="text-faint" aria-hidden />}
        <p className="mt-3 font-medium text-ink">{file ? file.name : "Drop an audio or video file here"}</p>
        <p className="mt-1 text-sm text-faint">
          {file ? `${(file.size / 1e6).toFixed(0)} MB. Click to choose a different file.` : "or click to choose one. MP4, M4A, WAV, MP3, WebM…"}
        </p>
        <input ref={input} type="file" accept={ACCEPT} className="sr-only" aria-label="Recording file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className={label}>Title
          <input name="title" className={field} placeholder="Weekly product sync" />
        </label>
        <label className={label}>Number of speakers
          <input name="num_speakers" type="number" min={1} max={20} className={field} placeholder="Detect automatically" />
        </label>
        <label className={`${label} sm:col-span-2`}>Attendees
          <input name="attendees" className={field} placeholder="Priya, Raj, Maya" />
          <span className="mt-1.5 block text-xs text-faint">
            Comma-separated. Helps spell names and limits which remembered voices are matched.
          </span>
        </label>
      </div>

      {error && <p role="alert" className="mt-4 text-sm text-danger">{error}</p>}
      <button type="submit" disabled={progress !== null}
        className="mt-5 w-full rounded-xl bg-blue px-4 py-3 font-semibold text-white shadow-[0_0_30px_rgb(46_75_255/0.45)] transition hover:brightness-110 disabled:opacity-60">
        {progress === null ? "Upload and take notes" : `Uploading ${Math.round(progress * 100)}%`}
      </button>
      <p className="mt-3 text-center text-xs text-faint">
        Only upload recordings everyone in them agreed to. Audio stays on this machine
        {config?.notes_llm === "claude" ? "; the transcript text is sent to Claude to write the notes."
          : config?.notes_llm === "local" ? `, and so does the transcript: ${modelLabel(config.extractor)} writes the notes here.`
          : "."}
      </p>
    </form>
  );
}
