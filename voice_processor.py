import os
import ssl
import urllib.request
import argparse
import json
import yaml
import time
import torch
import copy
import threading
import numpy as np
import soundfile as sf
import re
from pydub import AudioSegment

# F5-TTS imports
try:
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
    from f5_tts.model import DiT
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False

# Chatterbox TTS imports
try:
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError:
    CHATTERBOX_AVAILABLE = False

# -- Chatterbox top-level tuning constants (edit here, not via CLI args) --------
CHATTERBOX_EXAGGERATION = 0.5   # Expressiveness / emotion intensity  [0.0-1.0]
CHATTERBOX_CFG_WEIGHT   = 0.5   # Classifier-free guidance weight     [0.0-1.0]
CHATTERBOX_TEMPERATURE  = 0.8   # Sampling temperature                [0.0-1.0]
# -------------------------------------------------------------------------------

# Global lock for CUDA operations to prevent potential race conditions
CUDA_LOCK = threading.Lock()

# Force Python to create an unverified context globally
ssl._create_default_https_context = ssl._create_unverified_context

# HuggingFace environment block
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unicode status indicators
STATUS_ICONS = {
    "CLEAN":          "\u2705",
    "SLIGHT_SPEEDUP": "\U0001f536",
    "BIG_SPEEDUP":    "\U0001f534",
    "CRITICAL":       "\U0001f6a8",
}


def _segment_bounds_ms(seg):
    start_ms = max(0, int(round(seg["start"] * 1000)))
    end_ms = max(start_ms + 1, int(round(seg["end"] * 1000)))
    return start_ms, end_ms


def _audio_duration_seconds(audio):
    return len(audio) / 1000.0 if hasattr(audio, "__len__") else 0.0


def _speedup_status(speedup: float, max_speed: float) -> str:
    """Classify the speedup into a status string."""
    if speedup <= 1.0:
        return "CLEAN"
    elif speedup <= 1.2:
        return "SLIGHT_SPEEDUP"
    elif speedup < max_speed:
        return "BIG_SPEEDUP"
    else:
        return "CRITICAL"


def _time_stretch_audio(audio_chunk: AudioSegment, ratio: float) -> AudioSegment:
    """
    Speed-up / slow-down *audio_chunk* by *ratio* using pytsmod (preferred)
    or audiostretchy as fallback.
    ratio > 1 means speed-up (shorter output), < 1 means slow-down.
    Returns the modified AudioSegment.
    """
    samples = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_chunk.frame_rate
    channels = audio_chunk.channels

    if channels == 2:
        samples = samples.reshape(-1, 2).T   # (2, N)
    else:
        samples = samples[np.newaxis, :]     # (1, N)

    # Normalise to float32 [-1, 1]
    max_val = float(np.iinfo(audio_chunk.array_type).max)
    samples = samples / max_val

    try:
        import pytsmod as tsm
        # pytsmod expects (channels, samples) ndarray; returns same shape
        # pytsmod's `s` is a time-stretch factor: >1 = longer (slower).
        # Our `ratio` is a speed factor: >1 = shorter (faster).
        # Invert so the semantics match.
        stretched = tsm.wsola(samples, 1.0 / ratio)
    except ImportError:
        try:
            import audiostretchy.stretch as asts
            # audiostretchy also uses ratio > 1 = longer (slower).
            # Invert for the same reason as pytsmod above.
            inverted_ratio = 1.0 / ratio
            out_channels = []
            for ch in samples:
                out_channels.append(asts.stretch_array(ch, sample_rate, ratio=inverted_ratio))
            stretched = np.array(out_channels)
        except ImportError:
            raise RuntimeError(
                "Neither pytsmod nor audiostretchy is installed. "
                "Install one of them to enable fallback time-stretching: "
                "  pip install pytsmod"
            )

    # Back to int16
    stretched = np.clip(stretched * max_val, -max_val, max_val).astype(np.int16)

    if channels == 2:
        stretched = stretched.T.flatten()
    else:
        stretched = stretched.flatten()

    return audio_chunk._spawn(
        stretched.tobytes(),
        overrides={"frame_rate": sample_rate}
    )


def _fit_and_stretch(audio_chunk: AudioSegment, seg: dict, base_speed: float,
                     max_speed: float, max_duration_ms,
                     tts_supports_speed: bool):
    """
    Determine whether *audio_chunk* needs compression to fit its time slot.

    When tts_supports_speed=True the function returns the target speed so the
    caller can re-synthesise; the returned chunk is still the original.
    When tts_supports_speed=False the function time-stretches the chunk in place.

    Returns (final_chunk, effective_speedup, status_str).
    """
    start_ms, end_ms = _segment_bounds_ms(seg)
    slot_ms = max(1, end_ms - start_ms)
    limit_ms = max_duration_ms if max_duration_ms is not None else slot_ms
    gen_ms = len(audio_chunk)

    # The base_speed compresses the effective target window
    effective_limit_ms = max(1, int(limit_ms / base_speed))

    if gen_ms <= effective_limit_ms:
        speedup = base_speed
        status  = _speedup_status(speedup, max_speed)
        return audio_chunk, speedup, status

    # Need extra compression on top of base_speed
    ratio          = gen_ms / effective_limit_ms
    target_speedup = base_speed * ratio
    capped_speedup = min(target_speedup, max_speed)
    status         = _speedup_status(capped_speedup, max_speed)

    if tts_supports_speed:
        # Caller will re-synthesise; just return the current chunk + desired speed
        return audio_chunk, capped_speedup, status
    else:
        # Time-stretch to match capped_speedup
        stretched = _time_stretch_audio(audio_chunk, capped_speedup)
        return stretched, capped_speedup, status


def _trim_trailing_punctuation(text: str) -> str:
    """Trim trailing punctuation and whitespace from text."""
    if not isinstance(text, str):
        return text
    return re.sub(r"[\s\.,;:!?...]+$", "", text) + "  "


def _trim_to_time_slice(audio: AudioSegment, start_time_s: int = 0, end_time_s: int = None) -> AudioSegment:
    """Trim the audio segment using explicit time-start/time-end slice boundaries."""
    start_ms = max(0, int(start_time_s * 1000)) if start_time_s else 0
    if end_time_s is not None:
        end_ms = max(start_ms + 1, int(end_time_s * 1000))
        return audio[start_ms:end_ms]
    return audio[start_ms:]


def _init_synthesis_log():
    """Create a new synthesis log list and the target filename."""
    ts       = int(time.time())
    log_path = f"synthesis_log_{ts}.log"
    return [], log_path


def _log_segment(log, idx, seg, gen_len_s, slot_len_s, safe_slot_s,
                 speedup, status, text_fed):
    """Append a structured entry for one segment to the log list."""
    log.append({
        "segment":              idx + 1,
        "start_time_s":         round(seg["start"], 3),
        "generated_audio_len_s":round(gen_len_s, 3),
        "target_slot_len_s":    round(slot_len_s, 3),
        "safe_slot_with_gap_s": round(safe_slot_s, 3),
        "speedup_used":         round(speedup, 4),
        "status":               status,
        "text_fed":             text_fed,
    })


def _write_synthesis_log(log, log_path):
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.dump({"segments": log}, f, allow_unicode=True, sort_keys=False)


# -- Transcription --------------------------------------------------------------

def transcribe(input_file, language=None, model_size="base", time_start=0, time_end=None):
    print(f"🎤  Transcribing {input_file}...")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is required for transcription. "
            "Install it with: pip install faster-whisper"
        ) from exc

    audio         = AudioSegment.from_file(input_file)
    full_duration = len(audio) / 1000.0

    actual_end = time_end if time_end is not None else full_duration
    if time_start > 0 or time_end is not None:
        print(f"   Slicing audio: {time_start}s -> {actual_end}s")
        audio_slice    = audio[time_start * 1000 : int(actual_end * 1000)]
        temp_slice_path = "temp_whisper_slice.wav"
        audio_slice.export(temp_slice_path, format="wav")
        process_file   = temp_slice_path
    else:
        process_file = input_file

    compute_type  = "float16" if DEVICE == "cuda" else "int8"
    whisper_model = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)

    print(f"   Detecting speech (language={language if language else 'auto'})...")
    segments, info = whisper_model.transcribe(
        process_file,
        word_timestamps=True,
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=700,
            speech_pad_ms=400,
        )
    )
    print(f"   Detected language: {info.language} (p={info.language_probability:.2f})")

    transcribed_segments = []
    for segment in segments:
        transcribed_segments.append({
            "text":  segment.text.strip(),
            "start": float(segment.start) + time_start,
            "end":   float(segment.end)   + time_start,
        })

    if process_file == "temp_whisper_slice.wav" and os.path.exists("temp_whisper_slice.wav"):
        os.remove("temp_whisper_slice.wav")

    if not transcribed_segments:
        raise ValueError("No speech detected in the audio file slice.")

    return transcribed_segments, info.language


# -- Translation ----------------------------------------------------------------

def translate_segments_llm(segments, target_lang, source_lang="en", context=None):
    """Translates segments using Google Gemini LLM. Requires GOOGLE_API_KEY."""
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Skipping LLM translation.")
        return segments

    print(f"\U0001f310  LLM translation: {source_lang} -> {target_lang}...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    texts  = [seg["text"] for seg in segments]
    prompt = f"Translate the following list of strings from {source_lang} to {target_lang}.\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "Provide the translation as a JSON list of strings only, preserving the order and length.\n"
    prompt += json.dumps(texts, ensure_ascii=True, indent=2)

    with open("llm_translation_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    try:
        response      = model.generate_content(prompt)
        text_response = response.text

        if "```json" in text_response:
            text_response = text_response.split("```json")[1].split("```")[0].strip()
        elif "```" in text_response:
            text_response = text_response.split("```")[1].split("```")[0].strip()

        translated_texts = json.loads(text_response)

        if len(translated_texts) != len(segments):
            print(f"Warning: LLM returned {len(translated_texts)} segments, expected {len(segments)}.")
            return segments

        for seg, trans in zip(segments, translated_texts):
            seg["text"] = trans

    except Exception as e:
        print(f"Error during LLM translation: {e}")

    return segments


def translate_segments(segments, target_lang, source_lang="en", engine="nmt", context=None):
    """Translates segments using argostranslate (NMT) or LLM."""
    if engine == "llm":
        return translate_segments_llm(segments, target_lang, source_lang, context)

    import argostranslate.package
    import argostranslate.translate

    print(f"\U0001f310  NMT translation: {source_lang} -> {target_lang}...")

    try:
        argostranslate.package.update_package_index()
    except Exception as e:
        print(f"Warning: Could not update package index: {e}")

    available_packages   = argostranslate.package.get_available_packages()
    package_to_install   = next(
        filter(lambda x: x.from_code == source_lang and x.to_code == target_lang, available_packages),
        None
    )

    if package_to_install:
        print(f"   Installing translation package: {package_to_install}...")
        argostranslate.package.install_from_path(package_to_install.download())
    else:
        installed_packages = argostranslate.package.get_installed_packages()
        if not any(x.from_code == source_lang and x.to_code == target_lang for x in installed_packages):
            raise ValueError(f"No translation package found for {source_lang} -> {target_lang}")

    for seg in segments:
        original_text  = seg["text"]
        translated_text = argostranslate.translate.translate(original_text, source_lang, target_lang)
        seg["text"]    = translated_text

    return segments


# -- Data I/O ------------------------------------------------------------------

def save_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            return json.load(f)
        else:
            return yaml.safe_load(f)


# -- F5-TTS --------------------------------------------------------------------

def get_f5_model(repo_id="SWivid/F5-TTS", model_type="F5-TTS"):
    """
    Ensures F5-TTS model and vocab are available, downloading from HF if necessary.
    Returns (ckpt_path, vocab_path, model_cfg)
    """
    from huggingface_hub import hf_hub_download

    configs = {
        "F5-TTS": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        "E2-TTS": dict(dim=1024, depth=24, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    }

    paths = {
        "SWivid/F5-TTS": {
            "ckpt":  "F5TTS_v1_Base/model_1250000.safetensors",
            "vocab": "F5TTS_v1_Base/vocab.txt",
        },
        "SWivid/E2-TTS": {
            "ckpt":  "E2TTS_Base/model_1200000.safetensors",
            "vocab": "E2TTS_Base/vocab.txt",
        },
        "chosenek/f5-tts-czech-model": {
            "ckpt":  "model_last.pt",
            "vocab": "vocab.txt",
        },
    }

    config     = configs.get(model_type, configs["F5-TTS"])
    model_path = paths.get(repo_id, paths["SWivid/F5-TTS"])

    try:
        vocab_path = hf_hub_download(repo_id=repo_id, filename=model_path["vocab"])
        ckpt_path  = hf_hub_download(repo_id=repo_id, filename=model_path["ckpt"])
        return ckpt_path, vocab_path, config
    except Exception as e:
        raise RuntimeError(f"Could not fetch F5-TTS model: {e}")


def synthesize_f5(input_file, transcribed_segments, repo_id, model_type,
                  ref_audio=None, ref_text=None,
                  checkpoint_freq=0, target_duration_ms=None,
                  base_speed=1.0, max_speed=1.4):

    if not F5_TTS_AVAILABLE:
        raise ImportError("F5-TTS is not installed or available.")
    if not ref_audio:
        raise ValueError("F5-TTS requires --ref-audio-file for voice cloning.")

    ckpt_path, vocab_path, model_cfg = get_f5_model(repo_id, model_type)

    dit_model = load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path, device=DEVICE)
    vocoder   = load_vocoder()

    if target_duration_ms:
        final_timeline = AudioSegment.silent(duration=target_duration_ms)
    else:
        max_end        = max(seg["end"] for seg in transcribed_segments) if transcribed_segments else 0
        final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 2000)

    total    = len(transcribed_segments)
    log, log_path = _init_synthesis_log()

    for idx, seg in enumerate(transcribed_segments):
        current_start_ms, current_end_ms = _segment_bounds_ms(seg)
        slot_ms = max(1, current_end_ms - current_start_ms)

        if idx + 1 < total:
            next_start_ms, _ = _segment_bounds_ms(transcribed_segments[idx + 1])
            gap_ms           = max(0, next_start_ms - current_end_ms)
            max_duration_ms  = max(1, next_start_ms - current_start_ms)
        else:
            gap_ms          = 0
            max_duration_ms = None

        safe_slot_s = (max_duration_ms / 1000.0) if max_duration_ms else (slot_ms / 1000.0)

        # First pass at base_speed
        with CUDA_LOCK:
            wav_out, sr_out, _ = infer_process(
                ref_audio,
                ref_text if ref_text else seg["text"],
                seg["text"],
                dit_model,
                vocoder,
                device=DEVICE,
                speed=base_speed,
            )

        temp_gen_path   = f"temp_gen_f5_{idx}.wav"
        sf.write(temp_gen_path, wav_out, sr_out)
        generated_chunk = AudioSegment.from_wav(temp_gen_path)
        gen_len_ms      = len(generated_chunk)

        _, adjusted_speed, status = _fit_and_stretch(
            generated_chunk, seg,
            base_speed=base_speed, max_speed=max_speed,
            max_duration_ms=max_duration_ms,
            tts_supports_speed=True,
        )

        if adjusted_speed > base_speed:
            capped = min(adjusted_speed, max_speed)
            with CUDA_LOCK:
                wav_out, sr_out, _ = infer_process(
                    ref_audio,
                    ref_text if ref_text else seg["text"],
                    seg["text"],
                    dit_model,
                    vocoder,
                    device=DEVICE,
                    speed=capped,
                )
            sf.write(temp_gen_path, wav_out, sr_out)
            generated_chunk = AudioSegment.from_wav(temp_gen_path)
            adjusted_speed  = capped
            status          = _speedup_status(adjusted_speed, max_speed)

        icon = STATUS_ICONS[status]
        print(f"{icon} Segment {idx+1}/{total}  [{status}]  speed={adjusted_speed:.2f}x")

        _log_segment(log, idx, seg,
                     gen_len_s=gen_len_ms / 1000.0,
                     slot_len_s=slot_ms / 1000.0,
                     safe_slot_s=safe_slot_s,
                     speedup=adjusted_speed,
                     status=status,
                     text_fed=seg["text"])

        target_position_ms, _ = _segment_bounds_ms(seg)
        final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)

        if os.path.exists(temp_gen_path):
            os.remove(temp_gen_path)

        if checkpoint_freq > 0 and (idx + 1) % checkpoint_freq == 0:
            final_timeline.export(f"checkpoint_f5_{idx+1}.wav", format="wav")

    _write_synthesis_log(log, log_path)
    print(f"\U0001f4cb  Synthesis log saved -> {log_path}")
    return final_timeline


# -- Chatterbox TTS ------------------------------------------------------------

def synthesize_chatterbox(input_file, transcribed_segments,
                          ref_audio=None,
                          checkpoint_freq=0, target_duration_ms=None,
                          base_speed=1.0, max_speed=1.4, language=None):

    if not CHATTERBOX_AVAILABLE:
        raise ImportError(
            "Chatterbox TTS is not installed. Install with: pip install chatterbox-tts"
        )
    if not ref_audio:
        raise ValueError("Chatterbox TTS requires --ref-audio-file for voice cloning.")

    from chatterbox_git.src.chatterbox import mtl_tts  # to be able to use custom models like t3_cs.safetensors

    print(f"\U0001f5e3  Initialising Chatterbox TTS (device={DEVICE})...")
    model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model="t3_cs.safetensors")
    model.t3.to(DEVICE).eval()

    if target_duration_ms:
        final_timeline = AudioSegment.silent(duration=target_duration_ms)
    else:
        max_end        = max(seg["end"] for seg in transcribed_segments) if transcribed_segments else 0
        final_timeline = AudioSegment.silent(duration=int(max_end * 1000) + 2000)

    total    = len(transcribed_segments)
    log, log_path = _init_synthesis_log()

    for idx, seg in enumerate(transcribed_segments):
        current_start_ms, current_end_ms = _segment_bounds_ms(seg)
        slot_ms = max(1, current_end_ms - current_start_ms)

        if idx + 1 < total:
            next_start_ms, _ = _segment_bounds_ms(transcribed_segments[idx + 1])
            gap_ms           = max(0, next_start_ms - current_end_ms)
            max_duration_ms  = max(1, next_start_ms - current_start_ms)
        else:
            gap_ms          = 0
            max_duration_ms = None

        safe_slot_s   = (max_duration_ms / 1000.0) if max_duration_ms else (slot_ms / 1000.0)
        # text_to_speak = _trim_trailing_punctuation(seg.get("text", ""))
        text_to_speak = seg.get("text", "")

        if not text_to_speak.strip():
            silent_len      = max(100, max_duration_ms) if max_duration_ms else max(100, slot_ms)
            generated_chunk = AudioSegment.silent(duration=int(silent_len))
            target_pos, _   = _segment_bounds_ms(seg)
            final_timeline  = final_timeline.overlay(generated_chunk, position=target_pos)
            status          = "CLEAN"
            _log_segment(log, idx, seg,
                         gen_len_s=silent_len / 1000.0,
                         slot_len_s=slot_ms / 1000.0,
                         safe_slot_s=safe_slot_s,
                         speedup=base_speed, status=status, text_fed=text_to_speak)
            print(f"{STATUS_ICONS[status]} Segment {idx+1}/{total}  [{status}]  speed={base_speed:.2f}x  (silent)")
            continue

        temp_gen_path = f"temp_gen_chatterbox_{idx}.wav"

        with CUDA_LOCK:
            wav_tensor = model.generate(
                text_to_speak,
                language_id=language,
                audio_prompt_path=ref_audio,
                exaggeration=CHATTERBOX_EXAGGERATION,
                cfg_weight=CHATTERBOX_CFG_WEIGHT,
                temperature=CHATTERBOX_TEMPERATURE,
            )

        wav_np = wav_tensor.squeeze().cpu().numpy()
        sf.write(temp_gen_path, wav_np, model.sr)

        generated_chunk = AudioSegment.from_wav(temp_gen_path)
        gen_len_ms      = len(generated_chunk)

        # Chatterbox has no native speed param -> time-stretch
        stretched_chunk, adjusted_speed, status = _fit_and_stretch(
            generated_chunk, seg,
            base_speed=base_speed, max_speed=max_speed,
            max_duration_ms=max_duration_ms,
            tts_supports_speed=False,
        )

        icon = STATUS_ICONS[status]
        print(f"{icon} Segment {idx+1}/{total}  [{status}]  speed={adjusted_speed:.2f}x")

        _log_segment(log, idx, seg,
                     gen_len_s=gen_len_ms / 1000.0,
                     slot_len_s=slot_ms / 1000.0,
                     safe_slot_s=safe_slot_s,
                     speedup=adjusted_speed, status=status, text_fed=text_to_speak)

        target_pos, _  = _segment_bounds_ms(seg)
        final_timeline = final_timeline.overlay(stretched_chunk, position=target_pos)

        if os.path.exists(temp_gen_path):
            os.remove(temp_gen_path)

        if checkpoint_freq > 0 and (idx + 1) % checkpoint_freq == 0:
            final_timeline.export(f"checkpoint_chatterbox_{idx+1}.wav", format="wav")

    _write_synthesis_log(log, log_path)
    print(f"\U0001f4cb  Synthesis log saved -> {log_path}")
    return final_timeline


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate audio with Translation and Multiple TTS Engines"
    )
    parser.add_argument("--input-file",    type=str, help="Input WAV/audio file")
    parser.add_argument("--output-file",   type=str, default="regenerated_track.wav",
                        help="Output WAV file")
    parser.add_argument("--transcript-file", type=str, help="Transcript file (JSON/YAML)")

    # Language args
    parser.add_argument("--input-language",  type=str, help="Language of source audio")
    parser.add_argument("--output-language", type=str, help="Target language for translation")

    # Model configs
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper model size")
    parser.add_argument("--tts-engine",    type=str, default="f5-tts",
                        choices=["f5-tts", "chatterbox"], help="TTS engine to use")

    # Cloning / Ref args
    parser.add_argument("--ref-audio-file", type=str,
                        help="Reference audio file for voice cloning (required by all TTS engines)")
    parser.add_argument("--ref-text-file",  type=str,
                        help="Path to txt with text spoken in the reference audio (F5-TTS)")

    # F5-TTS specific args
    parser.add_argument("--f5-hf-repo",    type=str, default="SWivid/F5-TTS",
                        help="HuggingFace repo ID for F5-TTS")
    parser.add_argument("--f5-model-type", type=str, default="F5-TTS",
                        choices=["F5-TTS", "E2-TTS"], help="F5-TTS model variation")

    # Translation Engine args
    parser.add_argument("--translation-engine",       type=str, default="nmt",
                        choices=["nmt", "llm"], help="Translation engine to use")
    parser.add_argument("--translation-context-file", type=str,
                        help="Context file for LLM-based translation")

    # Speed args
    parser.add_argument("--base-speed", type=float, default=1.0,
                        help="Baseline synthesis speed multiplier (default: 1.0). "
                             "Increase when the target language uses more syllables.")
    parser.add_argument("--max-speed",  type=float, default=1.4,
                        help="Maximum allowed speedup ratio (default: 1.4). "
                             "Segments requiring more are marked CRITICAL.")

    parser.add_argument("--checkpoint-freq", type=int, default=1_000_000,
                        help="Save intermediate audio every N segments (0 to disable)")

    parser.add_argument("--trim-start-and-end-silence", action="store_true",
                        help="Trim leading and trailing silence from the final output")

    parser.add_argument("--time-start", type=int, default=0,
                        help="Start time in seconds for processing slice")
    parser.add_argument("--time-end",   type=int,
                        help="End time in seconds for processing slice")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--transcribe-only", action="store_true")
    mode_group.add_argument("--synthesize-only", action="store_true")
    mode_group.add_argument("--translate-only",  action="store_true")

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
        if not args.input_file:
            parser.error("--input-file is required unless --synthesize-only or --translate-only is used.")

    if not args.transcript_file:
        args.transcript_file = "transcription.yaml"

    print(f"\u2699\ufe0f  Device: {DEVICE}")

    # Load reference text if provided
    ref_text_content = None
    if args.ref_text_file:
        with open(args.ref_text_file, "r", encoding="utf-8") as f:
            ref_text_content = f.read().strip()

    # Load translation context if provided
    translation_context = None
    if args.translation_context_file:
        with open(args.translation_context_file, "r", encoding="utf-8") as f:
            translation_context = f.read().strip()

    if args.translate_only:
        transcribed_segments = load_data(args.transcript_file)

        if args.time_start > 0 or args.time_end is not None:
            transcribed_segments = [
                s for s in transcribed_segments
                if s["start"] >= args.time_start and
                   (args.time_end is None or s["end"] <= args.time_end)
            ]

        source_lang          = args.input_language if args.input_language else "en"
        transcribed_segments = translate_segments(
            transcribed_segments, args.output_language,
            source_lang=source_lang, engine=args.translation_engine,
            context=translation_context
        )

        base, ext      = os.path.splitext(args.transcript_file)
        output_filename = f"{base}_translated_to_{args.output_language}{ext}"
        save_data(transcribed_segments, output_filename)
        return

    transcribed_segments = None
    do_transcribe        = not args.synthesize_only
    do_synthesize        = not args.transcribe_only
    target_duration_ms   = None

    if do_transcribe:
        transcribed_segments, detected_lang = transcribe(
            args.input_file,
            language=args.input_language,
            model_size=args.whisper_model,
            time_start=args.time_start,
            time_end=args.time_end,
        )

        if args.input_file:
            audio              = AudioSegment.from_file(args.input_file)
            target_duration_ms = len(audio)

        if args.output_language:
            original_segments = copy.deepcopy(transcribed_segments)
            base, ext         = os.path.splitext(args.transcript_file)
            save_data(original_segments, f"{base}_original_lang{ext}")

            source_lang          = args.input_language if args.input_language else detected_lang
            transcribed_segments = translate_segments(
                transcribed_segments, args.output_language,
                source_lang=source_lang, engine=args.translation_engine,
                context=translation_context
            )

        save_data(transcribed_segments, args.transcript_file)

    if do_synthesize:
        if not transcribed_segments:
            transcribed_segments = load_data(args.transcript_file)
            # print loaded transcribed segments
            print(f"Loaded {len(transcribed_segments)} transcribed segments from {args.transcript_file}")

            if args.input_file and os.path.exists(args.input_file):
                audio              = AudioSegment.from_file(args.input_file)
                target_duration_ms = len(audio)
            else:
                max_end            = max(s["end"] for s in transcribed_segments) if transcribed_segments else 0
                target_duration_ms = int(max_end * 1000) + 2000

            if args.time_start > 0 or args.time_end is not None:
                transcribed_segments = [
                    s for s in transcribed_segments
                    if s["start"] >= args.time_start and
                       (args.time_end is None or s["end"] <= args.time_end)
                ]
                print(f"Filtered transcribed segments down to {len(transcribed_segments)} segments based on time slice {args.time_start}s -> {args.time_end}s")


        if args.tts_engine == "f5-tts":
            final_audio = synthesize_f5(
                args.input_file, transcribed_segments,
                repo_id=args.f5_hf_repo, model_type=args.f5_model_type,
                ref_audio=args.ref_audio_file, ref_text=ref_text_content,
                checkpoint_freq=args.checkpoint_freq,
                target_duration_ms=target_duration_ms,
                base_speed=args.base_speed,
                max_speed=args.max_speed,
            )
        elif args.tts_engine == "chatterbox":
            final_audio = synthesize_chatterbox(
                args.input_file, transcribed_segments,
                ref_audio=args.ref_audio_file,
                checkpoint_freq=args.checkpoint_freq,
                target_duration_ms=target_duration_ms,
                base_speed=args.base_speed,
                max_speed=args.max_speed,
                language=args.output_language,
            )

        if args.trim_start_and_end_silence:
            final_audio = _trim_to_time_slice(final_audio, args.time_start, args.time_end)

        final_audio.export(args.output_file, format="wav")
        print(f"\u2705  Output saved -> {args.output_file}")


if __name__ == "__main__":
    main()
