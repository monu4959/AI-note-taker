# Step 0 — Baseline on AMI

Hardware: RTX 4060 Laptop (8 GB), 16 GB RAM, Windows 11.

## Gate

**Passed.** Our pyannote community-1 DER on the AMI test split is **17.05%**, against the published
**17.0%** (same protocol: BUT "only_words" references, Mix-Headset audio, full-recording UEM,
no collar, overlap scored). This validates the diarization scoring code before anything is tuned.

## Results

| Metric | Value | Data | Tool |
|---|---|---|---|
| DER (diarization) | **17.05%** | 16 test meetings, 9.06 h | pyannote.metrics |
| JER (diarization) | 22.19% | 16 test meetings | pyannote.metrics |
| WER, speaker-agnostic (ORC-WER) | **25.8%** | 5 meetings, 21,547 ref words | meeteval |
| cpWER (who said what) | **29.4%** | 5 meetings | meeteval |
| ASR speed | 0.029 × real time (2.19 h in 224 s) | 5 meetings | faster-whisper |
| Diarization speed | 0.036 × real time (9.06 h in 1,173 s) | 16 meetings | pyannote |
| Peak GPU memory | ~2 GB ASR, 1.6 GB diarization | — | — |
| LLM summary | done, 14,849 input tokens (~$0.06 in; output count not recorded) | ES2004b | Claude Opus 5.5, effort high |

Per-meeting numbers: `artifacts/step0/scores.json`.

### Per meeting (transcription subset)

| Meeting | Type | WER | cpWER | DER |
|---|---|---|---|---|
| IS1009a | scenario kickoff | 21.5% | 27.2% | 19.7% |
| ES2004a | scenario kickoff | 21.5% | 23.8% | 18.8% |
| TS3003a | scenario kickoff | 17.8% | 21.8% | 17.1% |
| ES2004b | scenario, full length | 19.1% | 20.8% | 12.6% |
| EN2002a | non-scenario (natural) | 36.7% | 41.9% | 21.9% |

## Findings

1. **WER is dominated by deletions (80% of errors); substitutions are 3.9% of words.** When
   Whisper transcribes something it is usually right; it misses words. Fully-missed reference
   segments account for only 499 of 4,467 deleted words, so most deletions happen inside
   speech Whisper did transcribe. Two likely causes: overlapping speech in the single mixed
   channel (76.5% of reference words sit in segments overlapping another speaker), and Whisper's
   clean-verbatim style dropping repetitions and false starts that AMI's verbatim transcripts
   keep. The second cause is a hypothesis, not yet measured.
2. **Batched decoding is the right default.** On ES2004a: batched 21.5% WER in 26 s;
   sequential + VAD 22.6% in 167 s; sequential without VAD 24.2% in 257 s (and 10× the
   insertions). The decoding mode is not the cause of the deletions.
3. **Speaker attribution costs 3.6 WER points** (25.8% → 29.4%). Diarization errors are mostly
   missed speech (4.6–14.4%) rather than speaker confusion (1.6–6.2%), consistent with finding 1.
4. **pyannote over-counts speakers in 4 of 16 meetings** (5 found vs 4 real). Little DER cost,
   but Step 3 (speaker naming) must handle a spurious extra cluster.
5. **Word-to-speaker assignment is clean**: 3 of 16,543 words fell more than 0.5 s from any turn.
6. **Throughput leaves plenty of headroom**: a 1-hour meeting takes about 2 min ASR + 2 min
   diarization on the laptop GPU, inside the design's "≤ 5 min on one GPU" target.

7. **The one-call summary is usable as a baseline** (`artifacts/step0/summaries/ES2004b.md`).
   It found 9 decisions, 4 action items, 18 metrics and 8 open questions, each citing utterance
   IDs. Hand spot-check of 6 citations: all support their bullet, including the numbers
   (u0041 "about 5 metres", u0067 "75% … ugly", u0056 "30 to 40% of the buttons"). Speakers are
   still anonymous (SPEAKER_04 etc.) and nothing is machine-verified yet; both are Step 1/3 work.
   One metric came out without its unit ("50 million"); the model flagged it, which is the
   behaviour we want.

## Decisions taken

- **WhisperX not used.** It would downgrade torch 2.11 → 2.8 and huggingface-hub 2.4 → 0.36;
  it was only needed for forced alignment, and faster-whisper word timestamps assign cleanly
  (finding 5). Revisit if speaker-boundary errors show up in Step 1.
- **Reference text**: AMI manual annotations 1.6.2 with AMI's own segmentation; punctuation,
  truncated fragments (~1.3% of words), vocal sounds and disfluency markers excluded. Both
  sides normalized with Whisper's `EnglishTextNormalizer`.
- **Transcription subset**: the setup's `test.mini` (IS1009a, ES2004a, TS3003a) + EN2002a (only
  natural-meeting series) + ES2004b (full-length scenario meeting).

## Environment notes (Windows)

- faster-whisper needs cuBLAS/cuDNN: `pipeline/__init__.py` adds `torch/lib` to the DLL path.
- meeteval builds from source: MSVC Build Tools 2022 + `CL=/std:c++20`
  (`[tool.uv.extra-build-variables]` in `pyproject.toml`).
- Secrets live in a gitignored `.env`, loaded with priority over global environment variables.

## Reproduce

```sh
sh scripts/download_ami.sh
.venv/Scripts/python scripts/step0_diarize.py      # 16 meetings
.venv/Scripts/python scripts/step0_transcribe.py   # 5 meetings
.venv/Scripts/python scripts/step0_assign.py
.venv/Scripts/python scripts/step0_score.py
.venv/Scripts/python scripts/step0_summarize.py ES2004b
```
