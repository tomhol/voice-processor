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
# F5-TTS imports (keeping them here for now, but will make them optional if possible)
try:
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False

# Force Python to create an unverified context globally
ssl._create_default_https_context = ssl._create_unverified_context

# HuggingFace environment block
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def transcribe(input_file, language=None, model_size="base"):
    print(f"\n[Step 1] Loading Faster-Whisper to transcribe {input_file}...")
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

def translate_segments(segments, target_lang, source_lang="en"):
    """
    Translates segments using argostranslate (CTranslate2 NMT).
    """
    import argostranslate.package
    import argostranslate.translate

    print(f"\n[Translation] Preparing NMT model for {source_lang} -> {target_lang}...")
    
    # Update package index
    try:
        argostranslate.package.update_package_index()
    except Exception as e:
        print(f"Warning: Could not update package index: {e}")

    # Find and install package
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == source_lang and x.to_code == target_lang,
            available_packages
        ), None
    )

    if package_to_install:
        print(f"Installing translation package: {package_to_install}...")
        argostranslate.package.install_from_path(package_to_install.download())
    else:
        # Check if already installed
        installed_packages = argostranslate.package.get_installed_packages()
        if not any(x.from_code == source_lang and x.to_code == target_lang for x in installed_packages):
            raise ValueError(f"No translation package found for {source_lang} -> {target_lang}")

    print(f"Translating {len(segments)} segments...")
    for seg in segments:
        original_text = seg["text"]
        translated_text = argostranslate.translate.translate(original_text, source_lang, target_lang)
        seg["text"] = translated_text
        print(f"  {original_text} -> {translated_text}")
    
    return segments

def save_data(data, file_path):
    print(f"Saving transcription data to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
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

def synthesize_f5(input_file, transcribed_segments, tts_model_path, vocab_file_path, ref_audio=None, ref_text=None):
    if not F5_TTS_AVAILABLE:
        raise ImportError("F5-TTS is not installed or available.")
    
    print("\n[F5-TTS] Initializing Engine...")
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    dit_model = load_model(DiT, model_cfg, tts_model_path, vocab_file=vocab_file_path, device=DEVICE)
    vocoder = load_vocoder()

    base_audio = AudioSegment.from_file(input_file)
    max_end = max(seg["end"] for seg in transcribed_segments)
    final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 2000)

    for idx, seg in enumerate(transcribed_segments):
        print(f"Synthesizing section {idx+1}/{len(transcribed_segments)}...")
        
        temp_ref_path = f"temp_ref_{idx}.wav"
        current_ref_text = seg["text"] # Default to the segment text for self-cloning

        if ref_audio:
            # Global reference audio mode
            current_ref_path = ref_audio
            current_ref_text = ref_text if ref_text else seg["text"] 
        else:
            # Self-cloning mode (original behavior)
            ref_start_ms = int(seg["start"] * 1000)
            ref_end_ms = int(seg["end"] * 1000)
            base_audio[ref_start_ms:ref_end_ms].export(temp_ref_path, format="wav")
            current_ref_path = temp_ref_path

        wav_out, sr_out, _ = infer_process(current_ref_path, current_ref_text, seg["text"], dit_model, vocoder, device=DEVICE)
        
        temp_gen_path = f"temp_gen_{idx}.wav"
        sf.write(temp_gen_path, wav_out, sr_out)
        
        generated_chunk = AudioSegment.from_wav(temp_gen_path)
        target_position_ms = int(seg["start"] * 1000)
        final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
        
        if os.path.exists(temp_ref_path): os.remove(temp_ref_path)
        if os.path.exists(temp_gen_path): os.remove(temp_gen_path)

    return final_timeline

def synthesize_xtts(input_file, transcribed_segments, ref_audio=None, ref_text=None, language="en"):
    """
    Placeholder for XTTS v2 synthesis.
    Typically: 
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    """
    print(f"\n[XTTS] Initializing Engine (Language: {language})...")
    print("NOTE: XTTS implementation is a placeholder. Install 'TTS' library to use.")
    
    # Create a silent placeholder output
    max_end = max(seg["end"] for seg in transcribed_segments)
    return AudioSegment.silent(duration=int(max_end * 1000) + 1000)

def main():
    parser = argparse.ArgumentParser(description="Regenerate audio with Translation and Multiple TTS Engines")
    parser.add_argument("--input-file", type=str, required=True, help="Input noisy WAV file")
    parser.add_argument("--output-file", type=str, default="regenerated_track.wav", help="Output clean WAV file")
    parser.add_argument("--data-file", type=str, default="transcription.yaml", help="Data file (JSON/YAML)")
    
    # Language args
    parser.add_argument("--input-language", type=str, help="Language of source audio")
    parser.add_argument("--output-language", type=str, help="Target language for translation")
    
    # Model configs
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--tts-engine", type=str, default="f5-tts", choices=["f5-tts", "xtts"], help="TTS engine to use")
    
    # Cloning / Ref args
    parser.add_argument("--ref-audio-file", type=str, help="High-quality reference audio for voice cloning")
    parser.add_argument("--ref-text-file", type=str, help="Path to txt file containing text spoken in the reference audio")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transcribe-only", action="store_true")
    mode_group.add_argument("--synthesize-only", action="store_true")

    args = parser.parse_args()
    print(f"Using device: {DEVICE}")

    # Load reference text if provided
    ref_text_content = None
    if args.ref_text_file:
        print(f"Loading reference text from {args.ref_text_file}...")
        with open(args.ref_text_file, 'r', encoding='utf-8') as f:
            ref_text_content = f.read().strip()

    transcribed_segments = None
    do_transcribe = not args.synthesize_only
    do_synthesize = not args.transcribe_only

    if do_transcribe:
        transcribed_segments = transcribe(args.input_file, language=args.input_language, model_size=args.whisper_model)
        
        if args.output_language:
            transcribed_segments = translate_segments(transcribed_segments, args.output_language)
            
        save_data(transcribed_segments, args.data_file)
    
    if do_synthesize:
        if not transcribed_segments:
            transcribed_segments = load_data(args.data_file)
        
        if args.tts_engine == "f5-tts":
            # Defaults for F5-TTS
            TTS_MODEL = "ckpts/F5TTS_v1_Base/model_1250000.safetensors"
            VOCAB_FILE = "ckpts/F5TTS_v1_Base/vocab.txt"
            final_audio = synthesize_f5(args.input_file, transcribed_segments, TTS_MODEL, VOCAB_FILE, 
                                       ref_audio=args.ref_audio_file, ref_text=ref_text_content)
        elif args.tts_engine == "xtts":
            target_lang = args.output_language if args.output_language else (args.input_language if args.input_language else "en")
            final_audio = synthesize_xtts(args.input_file, transcribed_segments, 
                                         ref_audio=args.ref_audio_file, ref_text=ref_text_content, 
                                         language=target_lang)
        
        final_audio.export(args.output_file, format="wav")
        print(f"\n[Success] Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
