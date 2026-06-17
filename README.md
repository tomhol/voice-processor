# Audio Regenerator & Dubber

An AI-powered pipeline to transcribe, translate, and re-synthesize audio. This tool can be used to "clean" noisy speech by re-generating it with a high-quality TTS engine, or to perform automatic dubbing into other languages.

## Features

- **Robust Transcription:** Uses `faster-whisper` (including `large-v3` support) to extract text and precise timestamps from even low-quality audio.
- **Offline Translation:** Uses `argostranslate` (OpenNMT/CTranslate2) for private, offline translation between languages (e.g., English to Czech).
- **Advanced Synthesis:** 
  - **F5-TTS & E2-TTS:** Flow-matching engines for high-quality English synthesis.
  - **Auto-fetching Models:** Automatically downloads checkpoints from Hugging Face Hub if not found locally.
  - **Self-Cloning:** Can use the original noisy segments as a voice reference to preserve the speaker's tone while removing noise.
  - **Global Reference:** Provide a "Gold Standard" clean sample (`--ref-audio-file`) to completely re-voice the track.
  - **XTTS v2 Support:** Placeholder architecture for multi-lingual synthesis.
- **Modular Workflow:** Run only transcription, only synthesis, or the full pipeline.

## Installation

1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. 
## Usage

### 1. English Voice Restoration (Self-Cloning)
Transcribe noisy audio and re-synthesize it using the speaker's own voice (denoising):
```bash
python voice_processor.py --input-file noisy_track.wav --whisper-model large-v3
```

### 2. Voice Dubbing (Translation to Czech)
Translate English audio to Czech and save the transcription data:
```bash
python voice_processor.py --input-file noisy_track.wav --output-language cs --transcribe-only
```

### 3. Professional Re-voicing (Gold Standard)
Use a clean reference clip to dub the entire track:
```bash
python voice_processor.py --input-file noisy_track.wav \
    --ref-audio-file clean_sample.wav \
    --ref-text-file clean_sample.txt
```

### 4. Modular Mode (Synthesize from YAML)
Tune synthesis parameters without re-running the transcription. If doing voice cloning without `--input-file`, provide a global reference via `--ref-audio-file`:
```bash
python voice_processor.py --synthesize-only --transcript-file transcription.yaml \
    --ref-audio-file clean_sample.wav \
    --ref-text-file clean_sample.txt
```

### 5. Using E2-TTS Variation
Use the alternative E2-TTS model:
```bash
python voice_processor.py --input-file noisy_track.wav --f5-model-type E2-TTS
```

## Arguments

- `--input-file`: Path to source audio (required unless `--synthesize-only` is specified).
- `--output-file`: Output path (default: `regenerated_track.wav`).
- `--transcript-file`: Path to save/load metadata (default: `transcription.yaml`, mandatory when `--synthesize-only` is used).
- `--output-language`: Target language code (e.g., `cs`, `de`, `fr`).
- `--whisper-model`: Whisper model size (`base`, `small`, `medium`, `large-v3`).
- `--tts-engine`: Choose between `f5-tts` or `xtts`.
- `--ref-audio-file`: Path to a high-quality voice sample.
- `--ref-text-file`: Path to a `.txt` file containing the reference sample's text.
- `--f5-hf-repo`: HuggingFace repo ID (default: `SWivid/F5-TTS`).
- `--f5-model-type`: Choose `F5-TTS` or `E2-TTS`.

## Dependencies

- `faster-whisper`
- `f5-tts`
- `argostranslate`
- `pydub`
- `PyYAML`
- `huggingface_hub`
