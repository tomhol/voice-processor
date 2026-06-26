#!/usr/bin/env bash
set -euo pipefail

ENGINES=(
  "f5-tts"
  "chatterbox"
)

REF_FILES=(
  "en-E-Tavano.wav"
  "en-Jill-Engle-2.wav"
  "en-Jill-Engle.wav"
  "en-LJSpeech-female.wav"
  "en-M-Bertke.wav"
  "en-N-Prigoda.wav"
  "en-Story-Girl.wav"
  "gb-231.wav"
  "gb-261.wav"
  "gb-5.wav"
)

TRANSCRIPT_FILE="ruta_final_original_lang.yaml"
OUTPUT_LANGUAGE="en"
TIME_START=2686
TIME_END=2721

for ENGINE in "${ENGINES[@]}"; do
  for REF_FILE in "${REF_FILES[@]}"; do
    REF_BASENAME="${REF_FILE%.wav}"
    REF_TEXT_FILE="${REF_BASENAME}.txt"
    OUTPUT_FILE="${ENGINE}-${REF_BASENAME}-${TIME_START}.wav"

    python voice_processor.py \
      --synthesize-only \
      --tts-engine "${ENGINE}" \
      --transcript-file "${TRANSCRIPT_FILE}" \
      --ref-audio-file "${REF_FILE}" \
      --ref-text-file "${REF_TEXT_FILE}" \
      --output-language "${OUTPUT_LANGUAGE}" \
      --output-file "${OUTPUT_FILE}" \
      --time-start "${TIME_START}" \
      --time-end "${TIME_END}"
  done
done
