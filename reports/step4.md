# Step 4 — Free local notes (Qwen), polish and release gates

Date: 2026-11-19 (overnight). Goal: build and test without spending LLM money, with Claude
switchable back on later; every release gate checked by one script. **LLM spend: $0.00**
(the spend ledger is byte-identical to the start of the night; gate G0 checks this).

## Gate status (`scripts/gates.py`)

| Gate | Result | Detail |
|---|---|---|
| G0 $0 guard | **GREEN** | Claude blocked in code; spend ledger unchanged |
| G1 static | **GREEN** | ruff (pyflakes, syntax), TypeScript, web build |
| G2 unit | **GREEN** (56 passed) | |
| G3 integration | **GREEN** (18 passed) | API, recorder, recorder agent against the test database |
| G4 pipeline | **GREEN** (95 s; also stored in the test DB) | fresh 2-minute clip: Whisper + pyannote + Qwen, all citations valid |
| G5 UI end-to-end | **GREEN** (6 passed) | Playwright on Edge: list, meeting, notes, citation seeks audio, speakers, record page, phone width, no console errors |
| G6 quality | **BLOCKED** | no trustworthy free grader (below) |
| G7 speed | **GREEN** | 30-min meeting from scratch in 5.9 min (ratio 0.196, limit 0.25) |
| G8 package | **GREEN** | teammate recorder zip builds and starts from a fresh unzip |

Final run: `scripts/gates.py`, all gates, from a clean state; results in
`artifacts/gates/last_run.json`. 8 of 9 green; G6 is blocked, not failing on a number.

## Code review (independent reviewer agent) and fixes

The reviewer found 16 issues; the ones that mattered were fixed and each got a test or gate:

| Finding | Fix | Now checked by |
|---|---|---|
| **Blocker:** local notes could never be saved (`"2+local2"` into an int column): every real upload would fail after all the processing | store the base version in the column, full label in the payload | G3 regression test; G4 now stores real Qwen output in the test DB |
| Local checker's "unsupported" fell through to **verified** | becomes "Check this" | unit test |
| Upload size limits ran after the whole file was on disk; levels part read into RAM | size checked from Content-Length before reading (411/413); levels capped at 64 MB | G3 test |
| No Host/Origin checks: a web page could reach the server or recorder through your browser (DNS rebinding, cross-site POST to stop/delete a recording) | `services/security.py`: Host must be this machine / Tailscale; cross-site state changes refused; on server and recorder agent | G3 test |
| Ollama crash mid-meeting silently dropped windows and cached the result | only invalid replies are skipped; connection/server errors fail the stage (retried) | unit test; G4 fails on skipped windows |
| Long meetings: whole-transcript calls could exceed the context and be truncated silently | compact transcript for segmentation, name cues in 250-utterance chunks, too-long prompts refused | unit test |
| Eval caches could mix Claude and local results | model-tagged file names for every non-Opus/Sonnet run | review |
| `scripts/step0_summarize.py` called Claude directly | refuses unless `NOTES_LLM=claude` | review |
| Topic-span repair crashed on duplicate spans at the end; empty meetings | guarded | unit test |
| Notes citing only context lines duplicated neighbours | dropped | review |
| Firewall rule by address range also matches some hotel/mobile networks | rule tied to the Tailscale adapter (`-InterfaceAlias Tailscale`); `testclient` accepted only in tests | docs, G3 |
| Gate runner: G8 deleted a real bundle; G0 self-recorded a baseline; G0 didn't check the running server | temp build dir; baseline needs `--record-baseline`; G0 asks the live API its mode | gates |
| Leftover recorder process after G8 (uv's child) answered the next run | free port per run, whole process tree killed | G8 |

Not fixed (low): one HTTP client per call (fixed: now one client), a double-click could queue
two "write notes" jobs, unpinned dependencies in the teammate bundle.

## What changed

**Provider switch** (`pipeline/llm.py`, `.env`): `NOTES_LLM=local` (now) or `claude`.
- local: every LLM call goes to Qwen 2.5 7B through Ollama (`pipeline/llm_local.py`):
  JSON-schema constrained output, temperature 0, fixed seed, one retry with the validation
  error, calls logged to `artifacts/llm_local.jsonl` (the spend ledger is not touched).
- Claude is blocked twice: `llm.client()` and `llm.parse()` raise `ClaudeDisabled` for any
  Claude model, and the budget cap is set to $0 (`LLM_BUDGET_USD=0`). The API only offers the
  local model for "write notes with another model".
- Cached Claude results are never overwritten: Opus keeps the `insights` stage folder;
  local runs write `insights-qwen2.5-7b` and `names-qwen2.5-7b`; eval grades from a local
  model get model-tagged file names (`judge.j2-qwen2.5-7b.*`), Sonnet's stay `judge.j1.*`.

**Making a 7B model useful** (measured on AMI IS1008d; the Claude prompts are unchanged):

| Change | Why (measured) |
|---|---|
| Topics split into 40-utterance windows, each with 15 lines of read-only context | asked for everything per topic, Qwen found 1 of 5 human decisions and 2 of 4 actions |
| Two focused passes per window: decisions + actions, then key points / numbers / questions; cue phrases spelled out | decisions 1 -> 12, actions 2 -> 12 on the same meeting |
| Length caps in the JSON schema (6 citations per note, list and text lengths) | Qwen looped ("u0421, u0422, ...") until max tokens in 4 of 12 passes |
| A window whose reply never validates is skipped and recorded in the notes' meta | one bad window used to fail the whole meeting |
| The local checker can flag a note ("Check this"), never delete it; only code checks reject | Qwen called correct notes unsupported ("microphone on the side", which is in the human summary) |
| Verification in batches of 15 notes | one call with ~100 notes is unreliable for a 7B model |

**Memory** (16 GB laptop): at a 24k-token context Ollama's runner held ~15 GB of committed
RAM and the system ran out of memory (a background job was killed). Fixed: one fixed 12k
context (no model reloads between calls), local replies capped at 4,000 tokens, model
unloaded after each meeting's notes. During the 30-minute G7 run free RAM still dipped to
0.6 GB with the runner at 7.7 GB: close the screensaver/Discord/Chrome tabs on the server PC,
or use a smaller context if it happens again.

**UI** (Frontend Developer): footer "Notes: Qwen 2.5 7B on this computer" (or "Claude
(Opus 5.5)"); cost shows "free, written on this computer"; Claude is never offered while it
is off; processing estimate 10-15 min per hour locally; the privacy line on the record and
upload forms no longer says the transcript goes to Claude when it doesn't. New Playwright
tests (`tests/e2e`).

**Running for the team**: `scripts/start.ps1` waits for Docker Desktop and starts Ollama if
needed, `-NoBrowser` for unattended start; `scripts/install_autostart.ps1` (not installed:
your choice) starts the server 1 minute after you sign in.

## Quality: why G6 is blocked

There is no trustworthy free way to grade notes against the AMI human summaries tonight. The
AI Engineer built a local grader (BM25 pre-filter + one Qwen call per reference sentence) and
measured it against Sonnet's existing gradings of the same notes (dev meetings, 102
sentences): **lenient agreement 63.7%, kappa 0.08** (v1 one-call grader: kappa 0.13); it
never says "partial" and misses paraphrases even when the right note is in front of it. A
quality number from it would be meaningless, so the gate says BLOCKED instead of a number.

What we can say without a grader (IS1009c, 30 min, notes by each model on the same
transcript): Qwen kept 21 notes (5 decisions, 5 actions), Opus 44 (9 decisions, 5 actions,
5 numbers). Qwen's decisions cover the gist of all 3 human-listed decisions (push buttons,
batteries, voice recognition pending cost research) and include one overstatement ("will
have a voice recognition feature"). Good enough for building and testing; not a replacement
for Opus in real use.

**To measure it properly later** (notes stay free; only grading is paid, a few cents per
meeting): set `NOTES_LLM=claude` and a small budget, then
`scripts/step1_eval_insights.py --tag dev-qwen --extractor qwen2.5:7b IS1008d ES2011b TS3004c`
and `scripts/gates.py --only G6 --g6-tag dev-qwen`.

## Switching to Claude

In `.env`: `NOTES_LLM=claude` **and** a budget, because the cap is $0 now and the first Claude
call would be refused (e.g. `LLM_BUDGET_SCOPE=step4-claude`, `LLM_BUDGET_USD=5`); then
restart the API and worker (they read `.env` only at start). Opus notes, Sonnet checks and Sonnet
grading come back exactly as before (same stage folders, prompts and cache keys). G0 then
reports red by design (it checks that Claude is blocked); run the other gates with
`--skip G0`.

## Not done / known limits

- G6 has no number (above). The dev-meeting Qwen notes run was stopped by the system for low
  memory and, per its instruction, not restarted; IS1008d and IS1009c have Qwen notes.
- The local grader's cache-only rescore (`scripts/step4_calibrate_judge.py --versions 1 2
  --split dev`) was denied by the permission check when the AI Engineer tried it; it needs
  your OK (it makes no LLM calls).
- Qwen's notes are thinner than Opus's (about half as many) and occasionally overstate.
- The Mac recorder is still untested on a real Mac.
