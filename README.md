# Audio Regenerator & Dubber

An AI-powered pipeline to transcribe, translate, and re-synthesize audio. This tool can be used to "clean" noisy speech by re-generating it with a high-quality Text-to-Speech (TTS) engine, or to perform automatic dubbing into other languages.

## Features

- **Robust Transcription:** Uses `faster-whisper` to extract text and precise timestamps from audio.
- **Translation Options:**
  - **Offline NMT:** Uses `argostranslate` (OpenNMT/CTranslate2) for private, offline translation between languages.
  - **LLM Translation:** Uses Google Gemini (`gemini-2.5-flash`) for context-aware translations (requires `GOOGLE_API_KEY`).
- **Advanced Synthesis Engines:**
  - **F5-TTS & E2-TTS:** Flow-matching engines for high-quality speech synthesis. Automatically fetches models from Hugging Face Hub if not found locally.
  - **Chatterbox TTS:** A multilingual speech synthesis engine supporting custom models (e.g., `t3_cs.safetensors` for Czech).
  - **Voice Cloning:** Uses a high-quality sample (`--ref-audio-file`) to clone the voice of the target speaker.
- **Dynamic Speed Adaptation:** Automatically calculates if generated segments fit their designated time slots, applying time-stretching/speed-up dynamically to prevent overlaps. Supports `pytsmod` or `audiostretchy` for fallback time-stretching.
- **Modular Workflow:** Run only transcription, only translation, only synthesis, or the full end-to-end pipeline.
- **Audio Slicing & Cropping:** Process specific time ranges (`--time-start` and `--time-end`) or crop leading/trailing silence (`--crop`).

## Installation

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For time-stretching capabilities, ensure you have either `pytsmod` or `audiostretchy` installed. For Chatterbox TTS, ensure `chatterbox-tts` is installed.*

## Usage

### 1. Voice Dubbing with Offline NMT Translation (to Czech)
Translate English audio to Czech using Argos Translate and save the transcription data:
```bash
python voice_processor.py --input-file noisy_track.wav --output-language cs --transcribe-only
```

### 2. Voice Dubbing with LLM Translation (to Czech)
Translate English audio to Czech using Google Gemini with a context file:
```bash
export GOOGLE_API_KEY="your-api-key"
python voice_processor.py --input-file noisy_track.wav --output-language cs --translation-engine llm --translation-context-file translation-context.txt
```

### 3. Professional Re-voicing (Gold Standard Reference)
Use a clean reference clip to clone the voice and re-generate the entire track:
```bash
python voice_processor.py --input-file noisy_track.wav \
    --ref-audio-file clean_sample.wav \
    --ref-text-file clean_sample.txt
```

### 4. Modular Mode (Synthesize from YAML)
Re-synthesize audio from a pre-existing transcription YAML file (useful for iterative tuning of TTS parameters):
```bash
python voice_processor.py --synthesize-only --transcript-file transcription.yaml \
    --ref-audio-file clean_sample.wav \
    --ref-text-file clean_sample.txt
```

### 5. Using Chatterbox TTS
Use the Chatterbox multilingual TTS engine (e.g., for Czech):
```bash
python voice_processor.py --input-file noisy_track.wav \
    --tts-engine chatterbox \
    --ref-audio-file clean_sample.wav \
    --output-language cs
```

## Arguments

### Core Options
- `--input-file`: Path to source audio (required unless `--synthesize-only` or `--translate-only` is specified).
- `--output-file`: Output path (default: `regenerated_track.wav`).
- `--transcript-file`: Path to save/load metadata (default: `transcription.yaml`).

### Language & Translation Options
- `--input-language`: Language code of source audio (e.g., `en`, `cs`). If not provided for transcription, Whisper will attempt auto-detection.
- `--output-language`: Target language code for translation.
- `--translation-engine`: Choose between `nmt` (offline Argos Translate) or `llm` (Google Gemini, default: `nmt`).
- `--translation-context-file`: Path to a text file containing context to guide the LLM during translation.

### Transcription Options
- `--whisper-model`: Whisper model size (`base`, `small`, `medium`, `large-v3`, default: `base`).
- `--transcribe-only`: Run only transcription and exit.
- `--translate-only`: Run only translation on an existing transcript file and exit.
- `--synthesize-only`: Run only synthesis on an existing transcript file and exit.

### Synthesis Options
- `--tts-engine`: The synthesis engine to use (`f5-tts` or `chatterbox`, default: `f5-tts`).
- `--ref-audio-file`: Reference audio file for voice cloning (required by both TTS engines).
- `--ref-text-file`: Path to a `.txt` file containing the text spoken in the reference audio (used by F5-TTS).
- `--f5-hf-repo`: HuggingFace repo ID for F5-TTS (default: `SWivid/F5-TTS`).
- `--f5-model-type`: Choose `F5-TTS` or `E2-TTS` variation.
- `--checkpoint-freq`: Save intermediate audio every N segments (default: `100`, `0` to disable).

### Speed & Slicing Options
- `--base-speed`: Baseline synthesis speed multiplier (default: `1.0`). Increase if target language has more syllables.
- `--max-speed`: Maximum allowed speedup ratio (default: `1.4`). Segments requiring more speedup to fit are marked `CRITICAL`.
- `--time-start`: Start time in seconds for processing a slice of the audio (default: `0`).
- `--time-end`: End time in seconds for processing a slice of the audio.
- `--crop`: Trim leading and trailing silence from the final output using time slice boundaries.

---

## Utility Scripts

The project includes utility scripts to optimize and normalize transcripts in YAML format before synthesis.

### 1. `normalize_transcription.py`
Re-segments a YAML transcription file so that each output segment aligns with a natural sentence (split by periods `.`).
- **Splitting:** Timestamps for split segments are linearly interpolated by character count, and a configurable gap is inserted between them.
- **Merging:** Short segments that belong to the same sentence are merged.
- **Usage:**
  ```bash
  python normalize_transcription.py input.yaml -o output.yaml --max-words-in-segment 30 --gap-for-splitting 1.0
  ```

### 2. `optimize_transcription_for_tts.py`
Compacts a YAML transcription file by merging short segments. This helps TTS engines that struggle with very short clips.
- **Merging rules (in priority order):**
  1. Break at long gaps between segments (`--min-gap-for-break`, default 3s).
  2. Break at sentence-ending punctuation (`.`, `?`, `!`).
  3. Break at pause punctuation (`,`, `;`, `:`) when approaching the word limit.
  4. Break when segment would exceed `--max-words-in-segment`.
  5. Keep single oversized segments as-is.
- **Usage:**
  ```bash
  python optimize_transcription_for_tts.py input.yaml -o output.yaml --max-words-in-segment 20 --min-gap-for-break 3.0
  ```

### 3. `clean_yaml.py`
Filters out empty or noisy segments (e.g., Whisper hallucinated dot-only `.` segments) from a transcription YAML file.
- **Backups:** Automatically creates a backup of the original file (e.g. `transcription.yaml.bak`) before processing.
- **Usage:**
  ```bash
  python clean_yaml.py transcription.yaml
  ```
