import os

# # Replace this string with the actual path to your FFmpeg bin folder!
# os.add_dll_directory("C:/Users/tomhol/miniconda3/envs/vggsfm_tmp/Library/bin") 

import ssl
import urllib.request

# Force Python to create an unverified context globally
ssl._create_default_https_context = ssl._create_unverified_context

# HuggingFace environment block (Tells HF Hub explicitly to skip verifying)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

from faster_whisper import WhisperModel
from pydub import AudioSegment
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder
from f5_tts.model import DiT
import torch

# ==========================================
# CONFIGURATION
# ==========================================
AUDIO_INPUT = "noisy_track.wav"
OUTPUT_FINAL = "regenerated_track.wav"


# WHISPER_MODEL_SIZE = "base" # Change to "small" or "medium" for better quality if needed
WHISPER_MODEL_SIZE = "large-v3" 

#TTS_MODEL = "ckpts/F5TTS_v1_Base/model_1250000.safetensors"
#VOCAB_FILE = "ckpts/F5TTS_v1_Base/vocab.txt"
TTS_MODEL = "ckpts/F5TTS_czech/model_last.pt"
VOCAB_FILE = "ckpts/F5TTS_czech/vocab.txt"



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ==========================================
# STEP 1: TRANSCRIPTION WITH TIMESTAMPS
# ==========================================
print("\n[Step 1] Loading Faster-Whisper to transcribe noisy audio...")
# float16 is recommended for CUDA, float32 or int8 for CPU
compute_type = "float16" if DEVICE == "cuda" else "int8"
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=compute_type)

print("Processing audio segments and timings...")
segments, info = whisper_model.transcribe(AUDIO_INPUT, word_timestamps=True)

transcribed_segments = []
for segment in segments:
    # Capturing text and temporal boundaries for pacing reconstruction
    transcribed_segments.append({
        "text": segment.text.strip(),
        "start": segment.start,
        "end": segment.end
    })
    print(f"[{segment.start:05.2f}s -> {segment.end:05.2f}s]: {segment.text}")

if not transcribed_segments:
    raise ValueError("No speech detected in the audio file.")

# ==========================================
# STEP 2: LOAD F5-TTS EXPRESSIVE MODEL
# ==========================================
print("\n[Step 2] Initializing F5-TTS Flow-Matching Engine...")
# F5-TTS skips complex phoneme/duration alignments to natively map prosody/accents
model_cfg = dict(
    dim=1024, 
    depth=22, 
    heads=16, 
    ff_mult=2, 
    text_dim=512,    # Must be 512
    conv_layers=4,   # Must be 4
)
dit_model = load_model(DiT, model_cfg, TTS_MODEL, vocab_file=VOCAB_FILE, device=DEVICE)
vocoder = load_vocoder()

# ==========================================
# STEP 3: RE-SYNTHESIS & TIMING ALIGNMENT
# ==========================================
print("\n[Step 3] Regenerating audio blocks with original pacing and inflection...")
base_audio = AudioSegment.from_file(AUDIO_INPUT)
final_timeline = AudioSegment.silent(duration=int(transcribed_segments[-1]["end"] * 1000) + 1000)

for idx, seg in enumerate(transcribed_segments):
    print(f"Synthesizing section {idx+1}/{len(transcribed_segments)}...")
    
    # We pass the original noisy segment *as the voice clone reference* 
    # F5-TTS isolates the speaker's core vocal wavelength metrics, completely dropping background static
    ref_start_ms = int(seg["start"] * 1000)
    ref_end_ms = int(seg["end"] * 1000)
    temp_ref_path = f"temp_ref_{idx}.wav"
    
    # Extract reference clip
    base_audio[ref_start_ms:ref_end_ms].export(temp_ref_path, format="wav")
    
    # Generate clean audio chunk replicating accent & stress mapping
    # F5-TTS will generate an exportable raw numpy/wav array
    # wav_out, sr_out, _ = infer_process(temp_ref_path, seg["text"], seg["text"], dit_model, vocoder, device=DEVICE)
    wav_out, sr_out, _ = infer_process(
        # "liska.wav", "No a jednu si vzali stařeček a tů druhů panbů.", 
        "molavcova.wav", "Káťa si libovala, jaký je odtud výhled, ale Škubánek nemohl odtrhnout oči od šoféra.", 
        seg["text"], dit_model, vocoder, device=DEVICE, speed=1.2,
        )
    
    # Save the chunk temporarily
    temp_gen_path = f"temp_gen_{idx}.wav"
    import soundfile as sf
    sf.write(temp_gen_path, wav_out, sr_out)
    
    # Inject back onto a timeline matching the original timestamps
    generated_chunk = AudioSegment.from_wav(temp_gen_path)
    target_position_ms = int(seg["start"] * 1000)
    final_timeline = final_timeline.overlay(generated_chunk, position=target_position_ms)
    
    # Cleanup chunk files
    os.remove(temp_ref_path)
    os.remove(temp_gen_path)

# Export finalized clean file
final_timeline.export(OUTPUT_FINAL, format="wav")
print(f"\n[Success] File completely regenerated into a clean output: {OUTPUT_FINAL}")
