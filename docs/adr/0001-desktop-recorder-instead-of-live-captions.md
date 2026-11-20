# 1. Step 2 is a desktop recorder, not live captions

Status: accepted.

## Context

The design doc's Step 2 is live captions during a call (streaming ASR, a WebSocket, a latency
gate). What I actually need is for my own Zoom and Teams calls to get into the app without
downloading recordings by hand; captions during the call are not required. I mostly attend
meetings rather than host them, and use headphones only sometimes.

## Decision

Record on the computer of the person in the call, Granola-style: the default microphone (you) plus a loopback
of **every** output device (everyone else), each to its own 16 kHz track on one clock. When
the recording stops, the tracks are mixed and go through the existing upload pipeline.

- No bot: a meeting bot (Fireflies-style) needs per-platform integration (Zoom app review,
  no official Meet bot API) or a paid per-hour service, and is only worth it for meetings
  I don't attend.
- No host-only features: Zoom's per-participant audio files need the host to record.
- Calls are detected from Windows' microphone privacy record (Zoom/Teams holding the mic),
  which also stops the recording when the call ends. Recording always needs an explicit start
  with a consent checkbox (Massachusetts is an all-party-consent state).
- Without headphones the call leaks into the mic as echo; it is suppressed offline before
  mixing (`pipeline/recording.py`). The two tracks also identify the person recording: their
  voice is the one that is on the mic while the loopback is quiet.

## Consequences

- No captions during the call. The recorder already captures in 100 ms blocks, so streaming
  ASR can be added on top later without changing capture.
- Everything the computer plays is recorded (music, notifications). Windows 10 2004+ can
  capture a single process tree's audio (process loopback); not used yet.
- Each device has its own clock (measured: mic +450 ppm, HDMI -700 ppm), so writers lock
  their tracks to the system clock.
