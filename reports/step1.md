# Step 1 — Offline notetaker

Hardware: RTX 4060 Laptop (8 GB), 16 GB RAM, Windows 11. LLM spend for the whole step:
**$9.75** of the $15 cap I set for it.

## Gate

**Passed.** A recording uploaded in the browser (tested with an m4a of AMI IS1009a, 14 min,
and a 90 s wav through the actual upload form) is processed by the worker and shows a
speaker-labelled transcript and verified notes, with click-to-play citations. IS1009a went
from upload to notes in 2 min 14 s (transcribe 42 s, diarize 51 s, notes 31 s) for $0.175.

## What was built

| Part | Where | Notes |
|---|---|---|
| Cached stage runner | `pipeline/stages.py` | Key = stage version + params + upstream **output** fingerprints. A test caught that keying on params alone left downstream stale after a non-deterministic rerun. |
| Swappable ASR / diarizer | `pipeline/asr.py`, `pipeline/diarization.py` | One model on the GPU at a time. |
| Audio prep | `pipeline/audio.py` | Any format via PyAV (no system ffmpeg), 16 kHz mono, loudness normalized. |
| Cited insights | `pipeline/insights/` | Segment → per-topic extract (full transcript cached, ~95% cheaper reads) → verify → summarize from non-rejected items. |
| Verification | `pipeline/insights/verify.py`, `pipeline/numbers.py` | Deterministic checks + Sonnet 5.5 verdicts. Shared number matching so verification and eval agree. |
| Spend ledger + cap | `pipeline/llm.py` | Every call logged with cost; refused above the cap. |
| Eval harness | `evaluation/`, `scripts/step1_eval_insights.py` | AMI human summaries + summary→dialogue-act links as reference; Sonnet judge; cached. |
| DB, worker, API | `infra/`, `services/` | Postgres 17 + pgvector (port 5434), jobs table with SKIP LOCKED, FastAPI. |
| UI | `apps/web` | React + Vite + Tailwind; speaker ribbon doubles as scrubber; notes link to audio. |

Tests: 19 passing (stage caching, word assignment, verification checks, number matching,
API + job queue against a real Postgres test database).

## Insight evaluation

Reference: AMI's human abstractive summaries (decisions, actions, problems sections) and their
links to supporting dialogue acts. The 4 EN2002 test meetings have no summaries, so 12 test
meetings are used. Prompts were tuned on 3 dev meetings (IS1008d, ES2011b, TS3004c) and
frozen (prompt v2) before any test run.

| Metric | Dev (3) | **Test (12)** | ES2004 ours (4) | ES2004 human transcript (4) |
|---|---|---|---|---|
| Decisions recall, strict / lenient | 42.9 / 90.5% | **72.6 / 98.4%** (n=62) | 72.7 / 100% | 63.6 / 100% |
| Actions recall, strict / lenient | 50.0 / 100% | **70.0 / 85.0%** (n=20) | 50.0 / 75.0% | 75.0 / 100% |
| Problems recall, strict / lenient | 40.0 / 100% | **71.0 / 100%** (n=31) | 16.7 / 100% | 50.0 / 100% |
| Precision vs AMI (lower bound), decisions | 63.3% | 54.4% (n=79) | 42.9% | 38.1% |
| Precision vs AMI (lower bound), actions | 70.0% | 42.9% (n=42) | 16.7% | 21.4% |
| Citation agreement with human links | 70.8% | **70.9%** (220 pairs) | 68.6% | 59.7% |
| Wrong-number rate | 3.0% | **2.6%** (3/114) | 5.3% | **0.0%** |
| Items verified / needs review / rejected | 159/40/0 | 544/147/4 | 194/48/2 | 226/48/1 |

Strict = judge says "covered"; lenient adds "partial". Precision against AMI is a lower bound:
AMI summaries list a few sentences per section, so many correct notes have no counterpart.

### Hand-labelled precision and judge validation

Pending: `evaluation/labels/step1_labels.csv` (25 notes, 15 judge matches). Model labels are
kept separately in `step1_labels_model.json`; scoring will compare them once the sheet is filled.

## Findings

1. **Recall is high, strictness is the gap.** Almost no human decision is missed outright
   (98.4% lenient on test); about a quarter are only partly conveyed, mostly implicit-consensus
   decisions written up as proposals or open questions.
2. **Run-to-run variance is large on small sets.** Re-running the same prompt on the 3 dev
   meetings moved scores by 5–12 points (one item of 8 actions is 12.5 points). Prompt v1 vs
   v2 differences were mostly inside that band; v2 was kept for its two repeatable gains
   (action precision 47% → 70% in both runs; problems recall up).
3. **Wrong numbers come from transcription, and the pipeline flags them.** On the human
   transcript the wrong-number rate is 0%. Of the 3 test wrong numbers on ours, 2 were already
   marked "Check this" (a speaker's slip, "twenty" for €25; "5-6 designs"), and the 1 marked
   verified is most likely an eval-window artifact: Whisper heard "okay two", the human
   transcript has "Two okay" just outside the ±2 s window. On dev, the one real error (Whisper's
   "three" for "thirty") was also marked "Check this". No verified wrong number was confirmed.
4. **The number checker needed two fixes** before its numbers meant anything: spoken forms
   ("twelve fifty", "twelve euros fifty" for 12.50) and disfluent repeats ("eight eight
   million") were counted as wrong. 3 of 4 dev "wrong numbers" were checker errors.
5. **Transcript quality barely moved decision recall** (ES2004: 100% lenient on both), but
   cleaner text produced more items and no wrong numbers.
6. **Cost:** ~$0.46–0.65 per meeting hour (measured on TS3003b–d; earlier per-meeting costs are
   inflated because two concurrent runs shared the spend ledger — fixed by tagging each run).

## Known limitations (first work for the next step)

- **Spoken amounts can be misread and still verify.** "Twelve fifty" (€12.50) became a
  verified "1250 euro" metric on IS1009a: the literal digits match, so neither check catches
  it, and the human transcript says "twelve fifty" too. Fix in prompt v3 (normalize spoken
  amounts using context) plus a units-plausibility check.
- **Word boundaries at speaker changes.** Short fragments ("the", "I") land in the previous
  speaker's utterance. faster-whisper word timestamps are the cause; WhisperX-style forced
  alignment (deferred in Step 0 for dependency reasons) is the fix to measure with cpWER.
- **Summary can overstate.** A 90 s test clip's summary said the team "agreed" on something
  stored as a key point, not a decision. Summary verification is not yet done.
- **Speakers are anonymous** until renamed by hand; voice identification is Step 3.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```
Opens http://localhost:8000. Services: Postgres (Docker, :5434), API (:8000, also serves the
built web app), GPU worker. `-Dev` adds the Vite dev server on :5173.
