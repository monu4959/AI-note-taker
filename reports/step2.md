# Step 2 — Recording your calls on this computer

This replaces the design doc's "live captions" step
(`docs/adr/0001-desktop-recorder-instead-of-live-captions.md`). I mostly attend Zoom and
Teams calls rather than host them, and what I need is the transcript and notes after the
call, with or without headphones; captions during the call matter less. LLM spend for the step: **$0.50** of a $3 cap (only the one end-to-end run;
everything else is GPU/CPU).

## Gate

**Passed.** On a 39-minute AMI meeting recorded through the recorder's own code, 8 ways
(each of the 4 participants as "you", with and without headphones):

| | Target | Result |
|---|---|---|
| Transcript quality vs uploading the file | WER within 2 points | **worst +1.7** (7 of 8 better than upload) |
| Who is "you" | correct whenever named | **8 of 8** named, all correct |
| Audio kept on real devices | length = wall clock | **within ~80 ms over 5 min**, 0 s lost |
| Stop -> notes ready | <= 15% of meeting length | **12.9%** (5 min 3 s for 39 min) |

## How it works

```
Record ─► one thread per device, 16 kHz int16 files on disk every 100 ms   services/recorder.py
          mic-0.pcm      default microphone (follows Windows' default)
          others-N.pcm   loopback of EVERY output: headphones, speakers, HDMI, Bluetooth
          clock lock: each track nudged a sample at a time to the system clock
Stop   ─► meeting row at once; in the background:                           pipeline/recording.py
          echo detected? -> echo cancellation (adaptive filter, pipeline/aec.py)
                         -> residual echo suppression
          level both tracks, mix, soft limit -> source.wav + levels.npz
       ─► the normal upload pipeline (transcribe, diarize, names, notes)
       ─► "who is you": the one speaker whose voice dominates the mic           services/identify.py
```

| Part | Why it is built this way |
|---|---|
| Granola-style, no bot | A Fireflies-style bot needs Zoom app review / has no official Meet API, or a paid per-hour service, and only helps for meetings you don't attend. Zoom's per-person audio needs the host. |
| Every output device | The call may play on headphones, laptop speakers, a monitor or a Bluetooth headset, and you may plug headphones in mid-call; the monitor adds new devices within 2 s. |
| Clock lock | Measured on this laptop: the mic ran ~500 ppm fast and the HDMI output ~700 ppm slow against the system clock (mic 175 ms out after 5 min; 1.6 s per hour). Each writer resamples its 100 ms block by a few samples when it drifts more than 1.5 ms. After: every track within ~80 ms of wall time at 5 min (fixed start latency plus the last block), no growth. |
| Crash safety | Audio is fsync'd every second with `meta.json` beside it. A recording cut short by a crash or reboot is offered on the next start ("Take notes from it"). |
| Call detection | Windows' microphone privacy record (registry `CapabilityAccessManager`) shows which app holds the mic now. Zoom and Teams hold it only in a call, so: a "Record this call?" banner (and optional desktop notification), and auto-stop 30 s after the call app releases the mic. Never starts by itself: recording needs a click and a consent checkbox (Massachusetts is all-party consent). |
| Echo, two stages | Without headphones the mic hears the call ~60 ms later. Mixed naively, every remote word appears twice, and once the mic is levelled up to your voice the echo can be louder than the original. 1) Partitioned-block frequency-domain NLMS echo canceller with the loopback as reference (256 ms filter, adaptation frozen during double talk). 2) Frames where the mic still only carries echo are turned down 24 dB. Both are skipped when the mic doesn't follow the loopback (headphones). |
| "You" from the mic | Your voice is much louder on your mic than in the call audio; other people are the other way round. The cluster that clearly stands out (share >= 0.35 and >= 3x the runner-up) gets your name, method `mic`. In-person meetings (every voice on the mic) abstain. |
| Pause my mic | Zeros on the mic track while paused (the clock keeps running), for side conversations while you are muted in Zoom: muting in Zoom does not stop this recorder hearing you. |

UI: home page "Record a call" / "Upload a file"; the record form shows the mic and outputs it
will use; a recording bar on every page (timer, You / Others level meters, Pause my mic,
delete, Stop and take notes); banners for a detected call, an automatic stop and an
interrupted recording; Speakers tab explains a mic-named speaker. Live status comes over one
server-sent-events stream (no polling).

## Evaluation (`scripts/step2_eval_recorder.py`)

A real call has no reference transcript, so the two tracks are built from AMI ES2004b's
per-person headset channels: "you" = one headset, the call = the other three summed. Levels
are those of a laptop, not AMI's quiet close-talk headsets: your voice -26 dBFS, the call
-20 dBFS, mic noise -65 dBFS. "No headphones" adds the call to the mic 60 ms later, 10 dB
below the loopback (about as loud as your voice), with a 150 ms room tail. Each recording
is mixed by `pipeline/recording.py` and run through the same pipeline as an upload.

| You = | Headphones WER / cpWER | No headphones WER / cpWER | Echo removed (AEC median) | "You" share vs next |
|---|---|---|---|---|
| A (MEO015) | 17.7% / 20.5% | 17.8% / 21.7% | 19.9 dB | 0.45 / 0.07, 0.47 / 0.10 |
| B (FEE013) | 17.6% / 21.2% | 20.1% / 23.1% | 13.7 dB | 0.51 / 0.04, 0.52 / 0.08 |
| C (MEE014) | 17.0% / 21.0% | 16.6% / 20.7% | 7.5 dB | 0.68 / 0.10, 0.64 / 0.07 |
| D (FEE016) | 17.6% / 20.7% | 17.9% / 21.0% | 12.7 dB | 0.60 / 0.07, 0.58 / 0.07 |
| Upload (Mix-Headset file) | 18.4% / 20.8% | | | |

DER 13.6-15.2% vs 12.7% for the upload (the recordings are found to have 5 speakers, the
upload 6). The recordings beat the upload on WER because the per-person tracks are cleaner
than AMI's mix; the fair reading is "no worse", not "better".

What it took to get here (each found by this evaluation):
1. **"You" was never found** (0 of 2) with the first rule, "the call audio is silent while
   you talk": AMI's four people share a room, so the other headsets carry your voice too (as
   would someone in the room with you on a call). Rule changed to "your mic is clearly louder
   than the call" and a relative decision. 8 of 8.
2. **Without headphones, 2 of 4 lost >2 points** (22.4%, 21.3%), almost all deleted words:
   the mic is levelled up ~20 dB to your voice, so any echo the suppressor let through (double
   talk, room tail) landed louder than the remote voice. Fix: real echo cancellation before
   the suppressor, and a soft limiter.
3. **The first AEC run looked worse** (up to 35%): the simulation's noise floor (-60 dBFS)
   was as loud as AMI's quiet headset speech, which both stopped the filter adapting and got
   boosted into the mix; and the suppressor was switched off after cancellation. Fixed the
   simulation's levels (above) and always suppress the residual.

End-to-end in the app (no headphones, A as you, real worker job incl. Opus notes):
the recording was mixed (echo cancelled) and queued 42 s after Stop, and transcript, speakers,
names and notes were ready **303 s after Stop for 39 min of audio (12.9%)**. The app named
the right cluster "You" from the mic (share 0.46 vs 0.10 for the next speaker); WER 17.0%,
cpWER 20.7%. Notes: 96 items (77 verified, 19 to check), $0.47 Opus + Sonnet, plus $0.03
for transcript names. The meeting is in the app as "Simulated call: ES2004b".

## Known limitations

- **Everything the computer plays is recorded**: music, videos, notification sounds. Windows
  10 2004+ can capture one process tree's audio (process loopback); a next step.
- **Not yet tried on a real call.** The simulation is a linear echo path; real laptop
  speakers distort, and Teams/Zoom run their own echo cancellation on *their* mic stream (not
  ours). The first real calls, with and without headphones, are the real test.
- **One meeting, four "you"s.** 8 recordings of the same 39 minutes; the margin on the worst
  case (+1.7 of 2.0) is small.
- **The recorder must be running**: the API process holds it. Closing the API window mid-call
  keeps the audio so far (offered as "cut short"), but nothing after.
- **No captions during the call.** Capture already works in 100 ms blocks, so streaming ASR
  could be added on top.
- Bluetooth headsets switch to a low-quality "hands-free" profile when any app opens their
  mic; recording all outputs covers that endpoint, but the audio is 8-16 kHz telephone quality.

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1    # then "Record a call"
.venv/Scripts/python -m services.calls                          # is a Zoom/Teams call detected now?
.venv/Scripts/python scripts/step2_eval_recorder.py --you A     # simulated gate (GPU, no LLM)
.venv/Scripts/python scripts/step2_eval_recorder.py --you A --app   # + real app run with notes (~$0.40)
```
