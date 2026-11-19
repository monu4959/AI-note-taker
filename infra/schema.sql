-- Notetaker schema (design doc "Data model and API"). Idempotent: applied at API/worker start.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS meetings (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title         text NOT NULL,
    mode          text NOT NULL DEFAULT 'upload' CHECK (mode IN ('live', 'upload')),
    status        text NOT NULL DEFAULT 'queued'
                  CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
    attendees     jsonb NOT NULL DEFAULT '[]',
    num_speakers  int,
    source_name   text,
    audio_uris    jsonb NOT NULL DEFAULT '{}',
    duration_s    double precision,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    started_at    timestamptz,
    ended_at      timestamptz
);

CREATE TABLE IF NOT EXISTS persons (
    id            serial PRIMARY KEY,
    display_name  text NOT NULL,
    email         text
);

-- Step 3 (speaker memory). Embedding size of pyannote community-1 speaker embeddings.
CREATE TABLE IF NOT EXISTS voiceprints (
    id                serial PRIMARY KEY,
    person_id         int NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding         vector(256) NOT NULL,
    model_name        text NOT NULL,
    source_meeting_id uuid REFERENCES meetings(id) ON DELETE SET NULL,
    consented_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS speakers (
    id             serial PRIMARY KEY,
    meeting_id     uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    cluster_label  text NOT NULL,            -- SPEAKER_00 from diarization
    display_name   text,                     -- set by the user (or Step 3 identification)
    person_id      int REFERENCES persons(id) ON DELETE SET NULL,
    confidence     double precision,
    method         text CHECK (method IN ('mic', 'voiceprint', 'llm_cue', 'manual')),
    UNIQUE (meeting_id, cluster_label)
);

CREATE TABLE IF NOT EXISTS utterances (
    id              bigserial PRIMARY KEY,
    meeting_id      uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    seq             int NOT NULL,            -- u0142 <-> seq 142; insights cite these
    speaker_label   text NOT NULL,
    start_ms        int NOT NULL,
    end_ms          int NOT NULL,
    text            text NOT NULL,
    words           jsonb NOT NULL DEFAULT '[]',
    pass            text NOT NULL DEFAULT 'final' CHECK (pass IN ('live', 'final')),
    asr_confidence  double precision,
    UNIQUE (meeting_id, pass, seq)
);

CREATE TABLE IF NOT EXISTS insight_runs (
    id              serial PRIMARY KEY,
    meeting_id      uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    model           text NOT NULL,
    prompt_version  int NOT NULL,
    schema_version  int NOT NULL,
    payload         jsonb NOT NULL,          -- full insights output incl. meta and summary
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS insight_items (
    id             serial PRIMARY KEY,
    run_id         int NOT NULL REFERENCES insight_runs(id) ON DELETE CASCADE,
    item_key       text NOT NULL,            -- d1, a2, m3 ...
    type           text NOT NULL CHECK (type IN ('decision', 'action', 'metric',
                                                 'open_question', 'key_point', 'qa')),
    content        jsonb NOT NULL,
    evidence_seqs  int[] NOT NULL,
    status         text NOT NULL CHECK (status IN ('verified', 'needs_review', 'rejected')),
    UNIQUE (run_id, item_key)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           serial PRIMARY KEY,
    meeting_id   uuid NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    status       text NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'failed')),
    stage        text,
    progress     double precision NOT NULL DEFAULT 0,
    attempts     int NOT NULL DEFAULT 0,
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    finished_at  timestamptz,
    error        text
);
CREATE INDEX IF NOT EXISTS jobs_queued_idx ON jobs (created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS eval_runs (
    id          serial PRIMARY KEY,
    dataset     text NOT NULL,
    config      jsonb NOT NULL,
    metrics     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Step 3 additions (idempotent). A speaker row's name comes from one of: the user ('manual',
-- confirmed), a voiceprint match, or a transcript cue. `evidence` holds why it was proposed.
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS confirmed boolean NOT NULL DEFAULT false;
ALTER TABLE speakers ADD COLUMN IF NOT EXISTS evidence jsonb;
UPDATE speakers SET confirmed = true WHERE method = 'manual' AND NOT confirmed;
ALTER TABLE voiceprints ADD COLUMN IF NOT EXISTS source_label text;
ALTER TABLE voiceprints ADD COLUMN IF NOT EXISTS speech_s double precision;
CREATE UNIQUE INDEX IF NOT EXISTS persons_name_idx ON persons (lower(display_name));

-- Notes written by another model for an existing meeting: a job with {"extractor": ...} re-runs
-- only the notes stage (earlier stages are cached) and adds an insight run beside the others.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS params jsonb NOT NULL DEFAULT '{}';

-- Step 2 (desktop recorder): a recording made on this computer is mode 'live'; owner_name is
-- the person recording, who is named from the mic track (speakers.method = 'mic').
ALTER TABLE meetings ADD COLUMN IF NOT EXISTS owner_name text;

-- Notes v3: questions asked, with their answers, are their own note type.
ALTER TABLE insight_items DROP CONSTRAINT IF EXISTS insight_items_type_check;
ALTER TABLE insight_items ADD CONSTRAINT insight_items_type_check
    CHECK (type IN ('decision', 'action', 'metric', 'open_question', 'key_point', 'qa'));
