# AI notetaker

Meeting notetaker: record a Zoom/Teams call on this computer or upload a recording; get a
speaker-labelled transcript, cited notes by topic (with Q&A) and an Ask tab.
Design: `Meeting-Intelligence-Pipeline-Design.pdf`. Step-2 change from that design (desktop
recorder instead of live captions): `docs/adr/0001-desktop-recorder-instead-of-live-captions.md`.

## Layout

- `pipeline/`: audio prep, ASR (faster-whisper), diarization (pyannote), speaker naming,
  notes (`pipeline/insights/`: segment, extract, verify, summarize), meeting Q&A (`ask.py`)
- `services/`: FastAPI API, Postgres job queue and GPU worker, desktop recorder, teammate agent
- `apps/web/`: React + Vite UI, built into `apps/web/dist` and served by the API
- `evaluation/`, `scripts/`, `reports/`: AMI benchmarks, gates and one report per step

## Rules

- Secrets only in the gitignored `.env`; never print keys.
- LLM spend is capped per budget scope (`LLM_BUDGET_SCOPE`, `LLM_BUDGET_USD` in `.env`).
- Voiceprints are opt-in per person; never store one without consent.
- The repo is public: no real people's names or recordings in commits.

## Checks

```sh
.venv/Scripts/python scripts/gates.py          # every gate (static, unit, integration, UI, ...)
.venv/Scripts/python -m pytest tests --ignore=tests/e2e   # what CI runs
```
