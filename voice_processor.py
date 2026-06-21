import os
import ssl
import urllib.request
import argparse
import json
import yaml
import torch
import copy
import threading
import numpy as np
import soundfile as sf
import re
from faster_whisper import WhisperModel
from pydub import AudioSegment
# F5-TTS imports (keeping them here for now, but will make them optional if possible)
try:
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False

XTTS_TEMPERATURE = 0.65
XTTS_REPETITION_PENALTY = 5.0

# Global lock for CUDA operations to prevent potential race conditions
CUDA_LOCK = threading.Lock()

# Force Python to create an unverified context globally
ssl._create_default_https_context = ssl._create_unverified_context

# HuggingFace environment block
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _segment_bounds_ms(seg):
    start_ms = max(0, int(round(seg["start"] * 1000)))
    end_ms = max(start_ms + 1, int(round(seg["end"] * 1000)))
    return start_ms, end_ms


def _audio_duration_seconds(audio):
    return len(audio) / 1000.0 if hasattr(audio, "__len__") else 0.0


def _fit_audio_to_slot(audio_chunk, seg, speed_hint=1.0, max_duration_ms=None):
    start_ms, end_ms = _segment_bounds_ms(seg)
    slot_duration_ms = max(1, end_ms - start_ms)
    generated_duration_ms = len(audio_chunk)

    # Determine the actual available space (slot + gap to next)
    limit_ms = max_duration_ms if max_duration_ms is not None else slot_duration_ms

    if generated_duration_ms <= limit_ms:
        # Keep natural pace if it fits in the available space
        return audio_chunk, speed_hint

    # Speed up to fit into the limit
    ratio = generated_duration_ms / limit_ms
    new_speed = max(0.1, min(1.5, speed_hint * ratio))
    return audio_chunk, new_speed


def _trim_trailing_punctuation(text: str) -> str:
    """Trim trailing interpunction (punctuation) and whitespace from text.

    Removes trailing characters like . , ; : ? ! … and any trailing whitespace.
    """
    if not isinstance(text, str):
        return text
    # Remove trailing punctuation and whitespace.
    # Add ! to the end, the XTTS engine would stop any sound after that
    return re.sub(r"[\s\.,;:!?…]+$", "", text) + "  "  # hack for XTTS


def transcribe(input_file, language=None, model_size="base", time_start=0, time_end=None):
    print(f"\n[Step 1] Loading Faster-Whisper to transcribe {input_file}...")
    
    # Slice audio if needed
    audio = AudioSegment.from_file(input_file)
    full_duration = len(audio) / 1000.0
    
    actual_end = time_end if time_end is not None else full_duration
    if time_start > 0 or time_end is not None:
        print(f"Slicing audio: {time_start}s to {actual_end}s")
        audio_slice = audio[time_start * 1000 : int(actual_end * 1000)]
        temp_slice_path = "temp_whisper_slice.wav"
        audio_slice.export(temp_slice_path, format="wav")
        process_file = temp_slice_path
    else:
        process_file = input_file

    compute_type = "float16" if DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)

    print(f"Processing audio segments and timings (language={language if language else 'auto'})...")
    segments, info = whisper_model.transcribe(
        process_file, 
        word_timestamps=True, 
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=700, # Lowering this forces breaks on shorter pauses
            speech_pad_ms=400,
        )
    )
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    transcribed_segments = []
    for segment in segments:
        transcribed_segments.append({
            "text": segment.text.strip(),
            "start": float(segment.start) + time_start,
            "end": float(segment.end) + time_start
        })
        print(f"[{float(segment.start) + time_start:05.2f}s -> {float(segment.end) + time_start:05.2f}s]: {segment.text}")

    if process_file == "temp_whisper_slice.wav":
        if os.path.exists("temp_whisper_slice.wav"):
            os.remove("temp_whisper_slice.wav")

    if not transcribed_segments:
        raise ValueError("No speech detected in the audio file slice.")
    
    return transcribed_segments, info.language

def translate_segments_llm(segments, target_lang, source_lang="en", context=None):
    """
    Translates segments using Google Gemini LLM.
    Expects GOOGLE_API_KEY environment variable.
    """
    import google.generativeai as genai
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Skipping LLM translation.")
        return segments

    print(f"\n[Translation] Preparing LLM (Gemini) for {source_lang} -> {target_lang}...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Prepare the prompt
    texts = [seg["text"] for seg in segments]
    
    prompt = f"Translate the following list of strings from {source_lang} to {target_lang}.\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "Provide the translation as a JSON list of strings only, preserving the order and length.\n"
    prompt += json.dumps(texts, ensure_ascii=True, indent=2)
    
    print(f"Sending prompt to LLM (length: {len(prompt)} characters)...")
    # dump the prompt into a local file for debugging
    with open("llm_translation_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
        print("Prompt saved to llm_translation_prompt.txt for debugging.")

    try:
        response = model.generate_content(prompt)
        text_response = response.text
        
        # Strip markdown if present
        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        elif "```" in text_response:
             text_response = text_response.split("```")[1].split("```")[0].strip()
             
        translated_texts = json.loads(text_response)
        
        if len(translated_texts) != len(segments):
            print(f"Warning: LLM returned {len(translated_texts)} segments, but expected {len(segments)}.")
            return segments
            
        for seg, trans in zip(segments, translated_texts):
            seg["text"] = trans
            
    except Exception as e:
        print(f"Error during LLM translation: {e}")
    
    return segments

def translate_segments(segments, target_lang, source_lang="en", engine="nmt", context=None):
    """
    Translates segments using argostranslate (CTranslate2 NMT) or LLM.
    """
    if engine == "llm":
        return translate_segments_llm(segments, target_lang, source_lang, context)

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

def get_f5_model(repo_id="SWivid/F5-TTS", model_type="F5-TTS"):
    """
    Ensures F5-TTS model and vocab are available, downloading from HF if necessary.
    Returns (ckpt_path, vocab_path, model_cfg)
    """
    from huggingface_hub import hf_hub_download
    
    # Standard F5-TTS Base configuration
    # Note: If community models have different dims, we'd need to extend this mapping
    configs = {
        "F5-TTS": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        "E2-TTS": dict(dim=1024, depth=24, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    }
    
    paths = {
        "SWivid/F5-TTS": {
            "ckpt": "F5TTS_v1_Base/model_1250000.safetensors",
            "vocab": "F5TTS_v1_Base/vocab.txt"
        },
        "SWivid/E2-TTS": {
            "ckpt": "E2TTS_Base/model_1200000.safetensors",
            "vocab": "E2TTS_Base/vocab.txt"
        },
        "chosenek/f5-tts-czech-model": {
            "ckpt": "model_last.pt",
            "vocab": "vocab.txt"
        },
    }
    if model_type not in configs:
        print(f"Warning: Unknown model type '{model_type}', defaulting to F5-TTS config.")
        config = configs["F5-TTS"]
    else:
        config = configs[model_type]

    if repo_id not in paths:
        print(f"Warning: Unknown repo_id '{repo_id}', attempting to use local paths.")
        model_path = paths.get("SWivid/F5-TTS") if model_type == "F5-TTS" else paths.get("SWivid/E2-TTS")
    else:
        model_path = paths[repo_id]
    
    print(f"\n[F5-TTS] Resolving model '{model_type}' from {repo_id}...")
    
    try:
        vocab_path = hf_hub_download(repo_id=repo_id, filename=model_path["vocab"])
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=model_path["ckpt"])
        return ckpt_path, vocab_path, config
    except Exception as e:
        # Fallback to local check if HF is offline or path is different
        # local_ckpt = f"ckpts/{subfolder}/{filename}"
        # local_vocab = f"ckpts/{subfolder}/vocab.txt"
        # if os.path.exists(local_ckpt) and os.path.exists(local_vocab):
        #     print(f"HF Download failed, but found local files in {local_ckpt}. Using those.")
        #     return local_ckpt, local_vocab, config
        raise RuntimeError(f"Could not fetch F5-TTS model: {e}")

def synthesize_f5(input_file, transcribed_segments, repo_id, model_type, ref_audio=None, ref_text=None, checkpoint_freq=0, target_duration_ms=None):
    if not F5_TTS_AVAILABLE:
        raise ImportError("F5-TTS is not installed or available.")
    
    ckpt_path, vocab_path, model_cfg = get_f5_model(repo_id, model_type)
    
    print(f"[F5-TTS] Initializing {model_type} Engine...")
    dit_model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path, device=DEVICE)
    vocoder = load_vocoder()

    base_audio = AudioSegment.from_file(input_file) if input_file else None
    
    if target_duration_ms:
        final_timeline = AudioSegment.silent(duration=target_duration_ms)
    else:
        max_end = max(seg["end"] for seg in transcribed_segments) if transcribed_segments else 0
        final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 2000)

    current_ref_path = ref_audio
    for idx, seg in enumerate(transcribed_segments):
        # Calculate available space including gap to next segment
        current_start_ms, current_end_ms = _segment_bounds_ms(seg)
        gap_ms = 0
        if idx + 1 < len(transcribed_segments):
            next_start_ms, _ = _segment_bounds_ms(transcribed_segments[idx + 1])
            gap_ms = max(0, next_start_ms - current_end_ms)
            max_duration_ms = max(1, next_start_ms - current_start_ms)
        else:
            # For the last segment, we allow it to be slightly longer if needed, 
            # but still use its own duration as a hint for compression if it's way off.
            max_duration_ms = None 

        print(f"\n[F5-TTS] Synthesizing section {idx+1}/{len(transcribed_segments)}...")
        print(f"  Range: {seg['start']:.2f}s -> {seg['end']:.2f}s (len: {seg['end']-seg['start']:.2f}s)")
        if max_duration_ms:
            print(f"  Slot + Gap: {max_duration_ms/1000:.2f}s (Gap: {gap_ms/1000:.2f}s)")

        temp_ref_path = f"temp_ref_{idx}.wav"
        current_ref_text = seg["text"] # Default to the segment text for self-cloning

        current_ref_path = None
        if ref_audio:
            # Global reference audio mode
            current_ref_path = ref_audio
            current_ref_text = ref_text if ref_text else seg["text"] 
        else:
            # Self-cloning mode (original behavior)
            if not base_audio:
                raise ValueError("Self-cloning requires --input-file to extract reference audio segments.")
            ref_start_ms = int(seg["start"] * 1000)
            ref_end_ms = int(seg["end"] * 1000)
            base_audio[ref_start_ms:ref_end_ms].export(temp_ref_path, format="wav")
            current_ref_path = temp_ref_path

        speed = 1.0
        with CUDA_LOCK:
            wav_out, sr_out, _ = infer_process(
                current_ref_path,
                current_ref_text,
                seg["text"],
                dit_model,
                vocoder,
                device=DEVICE,
                speed=speed,
                # We remove fix_duration for first pass to avoid forced silence if text is too long
            )
        
        if wav_out is None or (isinstance(wav_out, np.ndarray) and wav_out.size > 0 and np.abs(wav_out).max() < 1e-5):
            print(f"  [Warning] F5-TTS produced silence for segment {idx+1}!")

        temp_gen_path = f"temp_gen_{idx}.wav"
        sf.write(temp_gen_path, wav_out, sr_out)
        
        generated_chunk = AudioSegment.from_wav(temp_gen_path)
        gen_len_ms = len(generated_chunk)
        
        generated_chunk, adjusted_speed = _fit_audio_to_slot(generated_chunk, seg, speed_hint=speed, max_duration_ms=max_duration_ms)
        
        limit_ms = max_duration_ms if max_duration_ms is not None else (current_end_ms - current_start_ms)
        is_compressing = adjusted_speed != speed
        
        print(f"  Generated length: {gen_len_ms/1000:.2f}s")
        if is_compressing:
            print(f"  [Compression] Required ratio: {adjusted_speed:.2f}x (to fit in {limit_ms/1000:.2f}s)")
            with CUDA_LOCK:
                wav_out, sr_out, _ = infer_process(
                    current_ref_path,
                    current_ref_text,
                    seg["text"],
                    dit_model,
                    vocoder,
                    device=DEVICE,
                    speed=adjusted_speed,
                )
            sf.write(temp_gen_path, wav_out, sr_out)
            generated_chunk = AudioSegment.from_wav(temp_gen_path)
            # Re-verify and potentially trim if still slightly over
            generated_chunk, _ = _fit_audio_to_slot(generated_chunk, seg, speed_hint=adjusted_speed, max_duration_ms=max_duration_ms)
        else:
            print(f"  [Compression] Not needed (fits in {limit_ms/1000:.2f}s)")
        
        target_position_ms, _ = _segment_bounds_ms(seg)
        final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
        
        if os.path.exists(temp_ref_path): os.remove(temp_ref_path)
        if os.path.exists(temp_gen_path): os.remove(temp_gen_path)

        # Checkpoint logic
        if checkpoint_freq > 0 and (idx + 1) % checkpoint_freq == 0:
            checkpoint_path = f"checkpoint_f5_{idx+1}.wav"
            print(f"  [Checkpoint] Saving intermediate audio to {checkpoint_path}...")
            final_timeline.export(checkpoint_path, format="wav")

    return final_timeline

def synthesize_xtts(input_file, transcribed_segments, ref_audio=None, ref_text=None, language="en", checkpoint_freq=0, target_duration_ms=None, debug=False):
    from TTS.api import TTS

    print(f"\n[XTTS] Initializing Engine (Language: {language})...")
    
    # Validate language code
    supported_langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja"]
    if language not in supported_langs:
        print(f"Warning: Language '{language}' might not be supported by XTTS. Supported: {supported_langs}")

    # Load model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if DEVICE == "cuda":
        tts.to(DEVICE)

    base_audio = AudioSegment.from_file(input_file) if input_file else None
    
    if target_duration_ms:
        final_timeline = AudioSegment.silent(duration=target_duration_ms)
    else:
        max_end = max(seg["end"] for seg in transcribed_segments) if transcribed_segments else 0
        final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 2000)

    for idx, seg in enumerate(transcribed_segments):
        # Calculate available space including gap to next segment
        current_start_ms, current_end_ms = _segment_bounds_ms(seg)
        gap_ms = 0
        if idx + 1 < len(transcribed_segments):
            next_start_ms, _ = _segment_bounds_ms(transcribed_segments[idx + 1])
            gap_ms = max(0, next_start_ms - current_end_ms)
            max_duration_ms = max(1, next_start_ms - current_start_ms)
        else:
            # For the last segment, we allow it to be slightly longer if needed
            max_duration_ms = None 

        print(f"\n[XTTS] Synthesizing section {idx+1}/{len(transcribed_segments)}...")
        print(f"  Range: {seg['start']:.2f}s -> {seg['end']:.2f}s (len: {seg['end']-seg['start']:.2f}s)")
        print(f"  Text: {seg['text']}")
        if max_duration_ms:
            print(f"  Slot + Gap: {max_duration_ms/1000:.2f}s (Gap: {gap_ms/1000:.2f}s)")

        # Prepare filenames for intermediate XTTS files. If debug is enabled, keep them in
        # an explicit directory; otherwise use temporary filenames and remove them.
        if debug:
            inter_dir = "xtts_intermediates"
            os.makedirs(inter_dir, exist_ok=True)
            temp_ref_path = os.path.join(inter_dir, f"ref_xtts_{idx}.wav")
            temp_gen_path_pre = os.path.join(inter_dir, f"gen_xtts_{idx}_speed1.00.wav")
            temp_gen_path_post = os.path.join(inter_dir, f"gen_xtts_{idx}_speedUPDATED.wav")
        else:
            temp_ref_path = f"temp_ref_xtts_{idx}.wav"
            temp_gen_path_pre = f"temp_gen_xtts_{idx}.wav"
            temp_gen_path_post = f"temp_gen_xtts_{idx}_speedUPDATED.wav"
        
        # Prepare text for XTTS synthesis (trim trailing interpunction)
        text_to_speak = _trim_trailing_punctuation(seg.get("text", ""))
        print(f"  Updated text: '{text_to_speak}'")

        # Determine current reference path: either the global ref_audio or an extracted slice
        if ref_audio:
            current_ref_path = ref_audio
        else:
            # Self-cloning mode: extract and optionally keep the reference slice
            if not base_audio:
                raise ValueError("Self-cloning requires --input-file to extract reference audio segments.")
            ref_start_ms = int(seg["start"] * 1000)
            ref_end_ms = int(seg["end"] * 1000)
            base_audio[ref_start_ms:ref_end_ms].export(temp_ref_path, format="wav")
            current_ref_path = temp_ref_path

        # If trimming leaves the segment empty (e.g., it was only punctuation), insert silence instead
        if not text_to_speak.strip():
            print(f"  [XTTS] Segment {idx+1} empty after trimming; inserting silence.")
            # Create a silent chunk approximately the slot length (or 100ms if unknown)
            silent_len = max(100, max_duration_ms) if max_duration_ms else max(100, current_end_ms - current_start_ms)
            generated_chunk = AudioSegment.silent(duration=int(silent_len))
            target_position_ms, _ = _segment_bounds_ms(seg)
            final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
            if not ref_audio and not debug and os.path.exists(temp_ref_path):
                os.remove(temp_ref_path)
            # No temp_gen_path created, continue to next segment
            continue

        # XTTS synthesis
        speed = 1.0
        with CUDA_LOCK:
            tts.tts_to_file(
                text=text_to_speak,
                speaker_wav=current_ref_path,
                language=language,
                file_path=temp_gen_path_pre,
                speed=speed,
                temperature=XTTS_TEMPERATURE,
                repetition_penalty=XTTS_REPETITION_PENALTY,
            )

        generated_chunk = AudioSegment.from_wav(temp_gen_path_pre)
        gen_len_ms = len(generated_chunk)
        
        generated_chunk, adjusted_speed = _fit_audio_to_slot(generated_chunk, seg, speed_hint=speed, max_duration_ms=max_duration_ms)
        
        limit_ms = max_duration_ms if max_duration_ms is not None else (current_end_ms - current_start_ms)
        is_compressing = adjusted_speed != speed
        
        print(f"  Generated length: {gen_len_ms/1000:.2f}s")
        if is_compressing:
            print(f"  [Compression] Required ratio: {adjusted_speed:.2f}x (to fit in {limit_ms/1000:.2f}s)")
            # Write post-speedup generation to a distinct filename including the speed value
            safe_speed = f"{adjusted_speed:.2f}".replace('.', '_')
            if debug:
                temp_gen_path_post = os.path.join(inter_dir, f"gen_xtts_{idx}_speed{safe_speed}.wav")
            else:
                temp_gen_path_post = f"temp_gen_xtts_{idx}_speed{safe_speed}.wav"
            with CUDA_LOCK:
                tts.tts_to_file(
                    text=text_to_speak,
                    speaker_wav=current_ref_path,
                    language=language,
                    file_path=temp_gen_path_post,
                    speed=adjusted_speed,
                    temperature=XTTS_TEMPERATURE,
                    repetition_penalty=XTTS_REPETITION_PENALTY,
                )
            generated_chunk = AudioSegment.from_wav(temp_gen_path_post)
            # Re-verify and potentially trim if still slightly over
            generated_chunk, _ = _fit_audio_to_slot(generated_chunk, seg, speed_hint=adjusted_speed, max_duration_ms=max_duration_ms)
        else:
            print(f"  [Compression] Not needed (fits in {limit_ms/1000:.2f}s)")
        
        target_position_ms, _ = _segment_bounds_ms(seg)
        final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
        
        # Preserve intermediate files only in debug mode; otherwise clean up temp files.
        if debug:
            if not ref_audio:
                print(f"  [Saved] Reference slice kept: {temp_ref_path}")
            print(f"  [Saved] Generated (pre/post) files: {temp_gen_path_pre} {temp_gen_path_post if is_compressing else ''}")
        else:
            # remove generated temp files used during synthesis
            if os.path.exists(temp_gen_path_pre):
                try:
                    os.remove(temp_gen_path_pre)
                except Exception:
                    pass
            if is_compressing and os.path.exists(temp_gen_path_post):
                try:
                    os.remove(temp_gen_path_post)
                except Exception:
                    pass

        # Checkpoint logic
        if checkpoint_freq > 0 and (idx + 1) % checkpoint_freq == 0:
            checkpoint_path = f"checkpoint_xtts_{idx+1}.wav"
            print(f"  [Checkpoint] Saving intermediate audio to {checkpoint_path}...")
            final_timeline.export(checkpoint_path, format="wav")

    return final_timeline

def main():
    parser = argparse.ArgumentParser(description="Regenerate audio with Translation and Multiple TTS Engines")
    parser.add_argument("--input-file", type=str, help="Input noisy WAV file")
    parser.add_argument("--output-file", type=str, default="regenerated_track.wav", help="Output clean WAV file")
    parser.add_argument("--transcript-file", type=str, help="Transcript file (JSON/YAML)")
    
    # Language args
    parser.add_argument("--input-language", type=str, help="Language of source audio")
    parser.add_argument("--output-language", type=str, help="Target language for translation")
    
    # Model configs
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--tts-engine", type=str, default="f5-tts", choices=["f5-tts", "xtts"], help="TTS engine to use")
    
    # Cloning / Ref args
    parser.add_argument("--ref-audio-file", type=str, help="High-quality reference audio for voice cloning")
    parser.add_argument("--ref-text-file", type=str, help="Path to txt file containing text spoken in the reference audio")

    # F5-TTS specific args
    parser.add_argument("--f5-hf-repo", type=str, default="SWivid/F5-TTS", help="HuggingFace repo ID for F5-TTS")
    parser.add_argument("--f5-model-type", type=str, default="F5-TTS", choices=["F5-TTS", "E2-TTS"], help="Model variation to use")

    # Translation Engine args
    parser.add_argument("--translation-engine", type=str, default="nmt", choices=["nmt", "llm"], help="Translation engine to use")
    parser.add_argument("--translation-context-file", type=str, help="Context for LLM-based translation")

    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000, help="Frequency (in segments) to save intermediate audio checkpoints (0 to disable)")

    parser.add_argument("--debug", action="store_true", help="Enable XTTS debug: save intermediate reference/gen WAV files")

    parser.add_argument("--time-start", type=int, default=0, help="Start time in seconds for processing slice")
    parser.add_argument("--time-end", type=int, help="End time in seconds for processing slice")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transcribe-only", action="store_true")
    mode_group.add_argument("--synthesize-only", action="store_true")
    mode_group.add_argument("--translate-only", action="store_true")

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.translate_only:
        if not args.transcript_file:
            parser.error("--transcript-file is mandatory when --translate-only is used.")
        if not args.output_language:
            parser.error("--output-language is mandatory when --translate-only is used.")
    elif args.synthesize_only:
        if not args.transcript_file:
            parser.error("--transcript-file is mandatory when --synthesize-only is used.")
    else:
        # Transcribe-only or full pipeline
        if not args.input_file:
            parser.error("--input-file is required unless --synthesize-only or --translate-only is used.")

    # Default transcript file if not specified
    if not args.transcript_file:
        args.transcript_file = "transcription.yaml"
    print(f"Using device: {DEVICE}")

    # Load reference text if provided
    ref_text_content = None
    if args.ref_text_file:
        print(f"Loading reference text from {args.ref_text_file}...")
        with open(args.ref_text_file, 'r', encoding='utf-8') as f:
            ref_text_content = f.read().strip()

    # Load translation context if provided
    translation_context = None
    if args.translation_context_file:
        print(f"Loading translation context from {args.translation_context_file}...")
        with open(args.translation_context_file, 'r', encoding='utf-8') as f:
            translation_context = f.read().strip()

    if args.translate_only:
        transcribed_segments = load_data(args.transcript_file)
        
        # Filter segments based on time slice
        if args.time_start > 0 or args.time_end is not None:
            print(f"Filtering segments for time slice: {args.time_start}s to {args.time_end}s")
            transcribed_segments = [s for s in transcribed_segments if s["start"] >= args.time_start and (args.time_end is None or s["end"] <= args.time_end)]

        source_lang = args.input_language if args.input_language else "en"
        
        transcribed_segments = translate_segments(
            transcribed_segments, 
            args.output_language, 
            source_lang=source_lang,
            engine=args.translation_engine,
            context=translation_context
        )
        
        # Auto-generate output filename
        base, ext = os.path.splitext(args.transcript_file)
        output_filename = f"{base}_translated_to_{args.output_language}{ext}"
        save_data(transcribed_segments, output_filename)
        return

    transcribed_segments = None
    do_transcribe = not args.synthesize_only
    do_synthesize = not args.transcribe_only
    target_duration_ms = None

    if do_transcribe:
        transcribed_segments, detected_lang = transcribe(
            args.input_file, 
            language=args.input_language, 
            model_size=args.whisper_model,
            time_start=args.time_start,
            time_end=args.time_end
        )
        
        # Determine target duration for synthesis
        if args.input_file:
            audio = AudioSegment.from_file(args.input_file)
            target_duration_ms = len(audio)

        if args.output_language:
            # Save original transcription first
            import copy
            original_segments = copy.deepcopy(transcribed_segments)
            base, ext = os.path.splitext(args.transcript_file)
            original_filename = f"{base}_original_lang{ext}"
            save_data(original_segments, original_filename)

            # Translate
            source_lang = args.input_language if args.input_language else detected_lang
            transcribed_segments = translate_segments(
                transcribed_segments, 
                args.output_language, 
                source_lang=source_lang,
                engine=args.translation_engine,
                context=translation_context
            )
            
        save_data(transcribed_segments, args.transcript_file)
    
    if do_synthesize:
        if not transcribed_segments:
            transcribed_segments = load_data(args.transcript_file)
            
            # Determine target duration from input file or full segments before filtering
            if args.input_file and os.path.exists(args.input_file):
                audio = AudioSegment.from_file(args.input_file)
                target_duration_ms = len(audio)
            else:
                max_end = max(s["end"] for s in transcribed_segments) if transcribed_segments else 0
                target_duration_ms = int(max_end * 1000) + 2000

            # Filter segments based on time slice
            if args.time_start > 0 or args.time_end is not None:
                print(f"Filtering segments for time slice: {args.time_start}s to {args.time_end}s")
                transcribed_segments = [s for s in transcribed_segments if s["start"] >= args.time_start and (args.time_end is None or s["end"] <= args.time_end)]

        if args.tts_engine == "f5-tts":
            final_audio = synthesize_f5(args.input_file, transcribed_segments, 
                                       repo_id=args.f5_hf_repo, model_type=args.f5_model_type,
                                       ref_audio=args.ref_audio_file, ref_text=ref_text_content,
                                       checkpoint_freq=args.checkpoint_freq,
                                       target_duration_ms=target_duration_ms)
        elif args.tts_engine == "xtts":
            target_lang = args.output_language if args.output_language else (args.input_language if args.input_language else "en")
            final_audio = synthesize_xtts(args.input_file, transcribed_segments, 
                                         ref_audio=args.ref_audio_file, ref_text=ref_text_content, 
                                         language=target_lang, checkpoint_freq=args.checkpoint_freq,
                                         target_duration_ms=target_duration_ms,
                                         debug=args.debug)
        
        final_audio.export(args.output_file, format="wav")
        print(f"\n[Success] Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()
