#!/bin/sh
# Download the AMI test split (16 meetings, Mix-Headset audio) and the manual annotations.
# Why Mix-Headset: it is the condition pyannote's published AMI DER (17.0%) is measured on.
# Why manual annotations 1.6.2: word-level transcripts with speakers (WER/cpWER reference),
# plus human summaries, decisions and actions used in later steps.
# Usage: sh scripts/download_ami.sh [MEETING ...]   (run from the project root; resumable)
#        no arguments = the 16 test meetings; e.g. `sh scripts/download_ami.sh IS1008d ES2011b`
set -e

ROOT="data/ami"
LIST="$ROOT/setup/lists/test.meetings.txt"
MIRROR="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

mkdir -p "$ROOT/audio" "$ROOT/annotations"

MEETINGS="$*"
[ -z "$MEETINGS" ] && MEETINGS=$(tr -d '\r' < "$LIST")
for m in $MEETINGS; do
  out="$ROOT/audio/$m.Mix-Headset.wav"
  echo "== $m"
  curl -fL --retry 3 -C - -o "$out" "$MIRROR/$m/audio/$m.Mix-Headset.wav"
done

zip="$ROOT/annotations/ami_public_manual_1.6.2.zip"
curl -fL --retry 3 -C - -o "$zip" "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
echo "done"
