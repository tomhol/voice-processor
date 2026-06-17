import os
import ssl
import urllib.request
import argparse
import json
import yaml
import torch
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
from f5_tts.model import DiT

# Force Python to create an unverified context globally
ssl._create_default_https_context = ssl._create_unverified_context

# HuggingFace environment block
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe(input_file, language=None, model_size="base"):
    print(f"\n[Step 1] Loading Faster-Whisper to transcribe {input_file}...")
    # float16 is recommended for CUDA, float32 or int8 for CPU
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)

    print(f"Processing audio segments and timings (language={language if language else 'auto'})...")
    segments, info = whisper_model.transcribe(input_file, word_timestamps=True, language=language)

    transcribed_segments = []
    for segment in segments:
        transcribed_segments.append({
            "text": segment.text.strip(),
            "start": float(segment.start),
            "end": float(segment.end)
        })
        print(f"[{segment.start:05.2f}s -> {segment.end:05.2f}s]: {segment.text}")

    if not transcribed_segments:
        raise ValueError("No speech detected in the audio file.")
    
    return transcribed_segments

def save_data(data, file_path):
    print(f"Saving transcription data to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            # Default to YAML
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

def load_data(file_path):
    print(f"Loading transcription data from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        else:
            return yaml.safe_load(f)

def synthesize(input_file, output_file, transcribed_segments, tts_model_path, vocab_file_path):
    print("\n[Step 2] Initializing F5-TTS Flow-Matching Engine...")
    model_cfg = dict(
        dim=1024, 
        depth=22, 
        heads=16, 
        ff_mult=2, 
        text_dim=512,
        conv_layers=4,
    )
    dit_model = load_model(DiT, model_cfg, tts_model_path, vocab_file=vocab_file_path, device=DEVICE)
    vocoder = load_vocoder()

    print("\n[Step 3] Regenerating audio blocks with original pacing and inflection...")
    base_audio = AudioSegment.from_file(input_file)
    
    # Calculate total duration needed
    max_end = max(seg["end"] for seg in transcribed_segments)
    final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 1000)

    for idx, seg in enumerate(transcribed_segments):
        print(f"Synthesizing section {idx+1}/{len(transcribed_segments)}...")
        
        ref_start_ms = int(seg["start"] * 1000)
        ref_end_ms = int(seg["end"] * 1000)
        temp_ref_path = f"temp_ref_{idx}.wav"
        
        base_audio[ref_start_ms:ref_end_ms].export(temp_ref_path, format="wav")
        
        # F5-TTS will generate an exportable raw numpy/wav array
        wav_out, sr_out, _ = infer_process(temp_ref_path, seg["text"], seg["text"], dit_model, vocoder, device=DEVICE)
        
        # Save the chunk temporarily
        temp_gen_path = f"temp_gen_{idx}.wav"
        sf.write(temp_gen_path, wav_out, sr_out)
        
        # Inject back onto a timeline matching the original timestamps
        generated_chunk = AudioSegment.from_wav(temp_gen_path)
        target_position_ms = int(seg["start"] * 1000)
        final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
        
        # Cleanup chunk files
        os.remove(temp_ref_path)
        os.remove(temp_gen_path)

    final_timeline.export(output_file, format="wav")
    print(f"\n[Success] File completely regenerated into a clean output: {output_file}")

def main():
    # Default values from original script
    WHISPER_MODEL_SIZE = "base"
    TTS_MODEL = "ckpts/F5TTS_v1_Base/model_1250000.safetensors"
    VOCAB_FILE = "ckpts/F5TTS_v1_Base/vocab.txt"

    parser = argparse.ArgumentParser(description="Regenerate audio using Whisper and F5-TTS")
    parser.add_argument("--input-file", type=str, help="Input noisy WAV file")
    parser.add_argument("--output-file", type=str, default="regenerated_track.wav", help="Output clean WAV file")
    parser.add_argument("--data-file", type=str, default="transcription.yaml", help="File to save/load transcription data (JSON or YAML)")
    parser.add_argument("--input-language", type=str, default=None, help="Language of the input audio (e.g., 'en', 'cs')")
    parser.add_argument("--whisper-model", type=str, default=WHISPER_MODEL_SIZE, help=f"Whisper model size (default: {WHISPER_MODEL_SIZE})")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transcribe-only", action="store_true", help="Only transcribe the audio and save to data-file")
    mode_group.add_argument("--synthesize-only", action="store_true", help="Only synthesize the audio using text from data-file")

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")

    if not args.input_file:
        # Input file is needed for transcription OR for reference voice clips during synthesis
        parser.error("--input-file is required")

    transcribed_segments = None

    # Determine what to do
    # If neither is set, do both.
    do_transcribe = not args.synthesize_only
    do_synthesize = not args.transcribe_only

    if do_transcribe:
        transcribed_segments = transcribe(args.input_file, language=args.input_language, model_size=args.whisper_model)
        # Always save data file if we transcribed
        save_data(transcribed_segments, args.data_file)
    
    if do_synthesize:
        if not transcribed_segments:
            transcribed_segments = load_data(args.data_file)
        
        synthesize(args.input_file, args.output_file, transcribed_segments, TTS_MODEL, VOCAB_FILE)

if __name__ == "__main__":
    main()
