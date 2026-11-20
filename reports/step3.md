# Step 3 — Speaker names

I built this before Step 2, which I later redesigned (`reports/step2.md`). LLM spend for the
step: **$1.07** of a $5 cap ($0.67 of it the AMI name-cue evaluation).

## Gate

**Passed: naming precision ≥ 95% across AMI meeting series.** Voice matching named 69 of 69
speakers correctly (100%; 95% confidence lower bound 94.8%) across 18 meetings, and named
**none** of 75 speakers it had never been given. In the app, remembering three voices from
AMI IS1009a and uploading IS1009b named all three by voice, with no hint, and left the
unremembered fourth person anonymous.

## How a speaker gets a name

```
diarization cluster (SPEAKER_03)
  1. named by you ................................ never changed automatically
  2. voice matches a remembered person >= 0.70 ... "Recognised by voice"      (precision-calibrated)
  3. introduced / addressed in the transcript .... "From the transcript, please confirm"
  4. otherwise ................................... "Speaker 4"  (abstain, never guess)
```

| Part | Where | Notes |
|---|---|---|
| Voiceprints | `pipeline/speaker_id.py` | The diarizer's per-cluster centroid (WeSpeaker ResNet34 inside pyannote community-1), already computed in Step 1, so naming adds no GPU time. Score = cosine to each person's best 3 voiceprints; Hungarian assignment so two speakers never share a name; clusters with < 10 s of speech are never named by voice. |
| Name cues | `pipeline/name_cues.py`, new cached `names` stage | Sonnet 5.5 proposes (speaker, name, kind, cited lines). Code keeps a cue only if the name is in a cited line, a self-introduction is said by that speaker, and an addressed speaker talks within 2 turns. A name claimed for two speakers, or a tie, abstains. ~$0.03 per meeting. |
| Combining | `services/identify.py` | Voice beats cue; a cue name already used in the meeting is dropped; your names are never touched. Runs after every job, and again for all meetings when you remember a new voice (cheap: no GPU, no LLM). |
| Consent | API + UI | Remembering a voice is an opt-in checkbox per person. "Remembered voices" lists everyone stored; Forget deletes every voiceprint and the names that came from it. The model name is stored with each voiceprint. |
| Notes keep labels | unchanged insights | Notes cite `SPEAKER_03` internally and the UI substitutes the current name, so a rename or a correction updates every action-item owner without re-running the LLM. |

Endpoints: `PATCH /api/meetings/{id}/speakers/{label}` (name, `remember_voice`),
`POST /api/meetings/{id}/identify`, `GET /api/people`, `DELETE /api/people/{id}`.

## Voiceprint calibration (`scripts/step3_calibrate.py`)

AMI scenario meetings come in series of four with the same four people. Six series (3 test:
IS1009, ES2004, TS3003; 3 dev: IS1008, ES2011, TS3004; 24 meetings): enroll everyone from
meeting a, identify them in b–d. Truth for a cluster is the AMI speaker covering most of its
speech (every one of the 75 test clusters with ≥ 10 s had a clear majority speaker).

| Threshold | Precision | Speakers named (of 75) | Never-enrolled person named |
|---|---|---|---|
| 0.50 | 100% | 92% | 2.7% (2/75) |
| 0.60 | 100% | 92% | 0% (closest call: 0.60) |
| **0.70 (chosen)** | **100% (69/69)** | **92%** | **0% (0/75)** |
| 0.80 | 100% | 88% | 0% |
| 0.90 | 100% | 72% | 0% |

Settings: "attendees" (only the series' 4 people are candidates) and "memory" (all 23
enrolled people are candidates) give identical results from 0.35 up. The stranger test
leaves each person out of enrollment in turn and checks their clusters stay unnamed.

Same person, later meeting: median similarity **0.93**, 5th percentile 0.74 (n=72).
Different people: median 0.08, 99th percentile 0.48, **max 0.62** (n=1653). 0.70 sits above
every different-person score observed, with room, and costs no coverage versus 0.50.

The 6 speakers never named:
- 3 are one ES2011 participant who could not be enrolled (the diarizer merged her into
  another speaker in meeting a). In b–d her voice scored up to 0.60 against a different
  woman: the real-world stranger case, and the reason the threshold is not 0.60.
- 3 are the right person scoring 0.51–0.57, all with only 14–37 s of speech. Abstaining is
  the intended outcome; they show as "Speaker N" with a one-click Name button.

So of the 72 speakers who were enrolled, 69 were named (95.8%), all correctly.

## Name cues (`scripts/step3_eval_cues.py`)

Ground truth from the **human** AMI transcript, independent of the pipeline: when a speaker
says "I'm Christine" there, that AMI speaker ID is Christine for the whole series (10 of the
24 people introduce themselves). Cues run on **our** transcript.

| | Names accepted | Right person | Exact name | Unchecked |
|---|---|---|---|---|
| Self-introduction ("I'm Betsy") | 12 | **8 / 8** | 7 / 8 | 4 |
| Addressed + reply ("Thanks, Raj") | 16 | 7 / 8 | 7 / 8 | 8 |
| Both kinds | 3 | 1 / 1 | 1 / 1 | 2 |
| **All** | **31** | **16 / 17 (94%)** | 15 / 17 (88%) | 14 |

- The one wrong person: "…is that right, Mike?" followed by a *different* speaker's
  "Shall I start?". The reply rule assumes the next speaker is the one addressed.
- The one wrong spelling is the right person: Whisper heard "I'm Janvi" where the human
  transcript has "Jeanne". No check can fix a misheard name, which is why cue names ask for
  confirmation.
- The 14 unchecked names are at least consistent: within a series no name ever landed on two
  different people, and repeated names ("Ed" in all four IS1008 meetings, "William", "Paul")
  always landed on the same person.

Cue names are therefore shown as proposals ("From the transcript, please confirm", with the
quoted line playable), not as facts, and never override a voice match. That matches the
design: cues count at lower confidence.

## Real meeting (a 64-minute Zoom class call I took part in, 12 speakers)

The cue stage proposed 8 names, all from hand-offs and direct address ("<name>, you are the
group leader?" → "Yeah, professor"; "<name> will be carrying around for the ..." → next
presenter). I checked each against its lines; all are plausible, but only the participants
can confirm names and spellings (several were ambiguous, e.g. two spellings of one name, or a
name the recognizer cut short). Participants' names are left out of this report. The professor and the
two people who talked most were left unnamed: nobody said their names in a checkable way.
$0.045.

## Notes model: Opus 5.5 vs Sonnet 5.5

Any meeting can now get a second set of notes from the other model (Notes tab: "Also write
notes with Sonnet 5.5", then "Compare side by side"). Only the LLM steps re-run; the
verifier is Sonnet in both cases.

AMI dev meetings (IS1008d, ES2011b, TS3004c). Two Opus runs of the same prompt are shown,
because run-to-run noise on 3 meetings is 5–12 points:

| | Opus run 1 | Opus run 2 | Sonnet |
|---|---|---|---|
| Decisions recall, strict / lenient (n=21) | 43 / 90% | 48 / 100% | 57 / 90% |
| Actions recall, strict / lenient (n=8) | 50 / 100% | 38 / 100% | 25 / 88% |
| Problems recall, strict / lenient (n=5) | 40 / 100% | 60 / 80% | 60 / 100% |
| Decisions written | 30 | 30 | **22** |
| Actions written | 10 | 10 | **6** |
| Citation agreement with human links | 71% | 64% | 76% |
| Wrong-number rate | 3% | 6% | 0% |
| Cost, 3 meetings | $1.14 | $0.92 | **$0.65** |

User's Zoom call (64 min): Opus $0.57, 116 notes (13 decisions, 17 actions); Sonnet
$0.33, 96 notes (6 decisions, 7 actions). Sonnet's decisions were all valid, but it left
out several that Opus caught (e.g. rejecting Careerjet for its 500-character cap).

Reading: per-note accuracy and grounding are as good or better with Sonnet, at ~40% lower
cost, but **Sonnet writes far fewer decisions and action items**, the notes users rely on
most; on AMI that cost one of the 8 human action items. Opus stays the default. The judge
is Sonnet, so it is scoring its own model's notes here; treat Sonnet's small wins with
that in mind.

## Known limitations

- **Threshold is calibrated on AMI headset-mix audio.** Zoom/Meet audio (codecs, laptop mics,
  echo) may shift similarities. Re-check on a few consented real calls before trusting 0.70
  there; the stranger test is the one to repeat.
- **The diarizer sometimes splits one person into two clusters** (Betty and "Betsy" in
  IS1009b). Both get the right name here, but the short half often has too little speech
  to be named by voice. Merging same-name clusters is a possible next step.
- **Small sample:** 69 named speakers, 24 people. The confidence bound (94.8%) is just under
  the 95% target, even though no error was observed.
- **One voiceprint per meeting** comes from the diarizer's centroid. The design suggests
  several clean-segment embeddings per person; the centroid was enough to pass the gate, so
  it was not added.
- Persons are matched by name: two different people called "Raj" would share a voice profile
  if both are remembered under the same name.

## UI redesign

The whole app was restyled after moninotes.com: a dark near-black ground with an
electric-blue glow, Inter throughout, and a three-pane app shell (icon rail, meeting list
with search, meeting pane). The player became a custom bar (play, time, speed) with the
speaker timeline as the scrubber. New screens: **Speakers** tab (proposed names with their
evidence, voice sample, Confirm / Rename, opt-in "Remember this voice") and **Remembered
voices**. Amber is reserved for "Check this" and transcript-proposed names, so it always
means "look here". Checked at 1440 px and 390 px (no horizontal scroll, no console errors).

## Run it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
.venv/Scripts/python scripts/step3_calibrate.py     # voiceprint threshold sweep (no LLM)
.venv/Scripts/python scripts/step3_eval_cues.py     # name-cue precision (cached LLM calls)
```
