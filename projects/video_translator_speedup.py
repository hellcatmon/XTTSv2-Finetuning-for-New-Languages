import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize
import json
import sys
import os
from pydub import AudioSegment
import torch.multiprocessing as mp
from itertools import cycle

sys.path.append('/content/XTTSv2-Finetuning-for-New-Languages')

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# -----------------------------------------------
# 7. Generate TTS translation

# -----------------------------------------------
# 7. Generate TTS translation config
adjust_translated_audio_duration_to_original = True

# Get available GPUs
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    device = "cpu"
    devices = ["cpu"]
else:
    devices = [f"cuda:{i}" for i in range(num_gpus)]

# Model paths
xtts_checkpoint = "/content/checkpoints/XTTS_v2.0_original_model_files/model.pth"
xtts_config = "/content/checkpoints/XTTS_v2.0_original_model_files/config.json"
xtts_vocab = "/content/checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model on each GPU
models = {}
for device in devices:
    config = XttsConfig()
    config.load_json(xtts_config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    model.to(device)
    model.eval()
    models[device] = model

# Create output directory if it doesn't exist
output_dir = "/content/out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load segments from JSON file
files_json_path = '/content/downloads/audio_chunks/filenames.json'

with open(files_json_path, 'r', encoding='utf-8') as f:
    segments = json.load(f)

# Extract filenames for reference audio
filenames = segments

output_files = []

# Create device cycle for round-robin assignment
device_cycle = cycle(devices)

# Generate TTS for each segment
with torch.no_grad():  # Disable gradient computation for inference
    for i, segment in enumerate(tqdm(segments, desc="Generating TTS")):
        # Get next device in rotation
        current_device = next(device_cycle)
        current_model = models[current_device]
        
        # Get corresponding audio chunk
        speaker_audio_file = filenames[i]['filename']
        print(f"Processing on {current_device}: {speaker_audio_file}")

        # Get latent and embedding for TTS
        gpt_cond_latent, speaker_embedding = current_model.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=current_model.config.gpt_cond_len,
            max_ref_length=current_model.config.max_ref_len,
            sound_norm_refs=current_model.config.sound_norm_refs,
        )

        # Generate TTS audio
        wav_chunk = current_model.inference(
            text=segment['translated_text'],
            language="be",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.1,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=10,
            top_p=0.3,
        )

        # Adjust audio duration if requested
        if adjust_translated_audio_duration_to_original:
            # Calculate target duration from original segment
            target_duration = segment['end'] - segment['start']

            # Get current duration in seconds
            current_duration = len(wav_chunk["wav"]) / current_model.config.audio.sample_rate

            # Calculate speed ratio needed
            speed_ratio = current_duration / target_duration

            # Resample audio to match target duration
            wav_resampled = torch.nn.functional.interpolate(
                torch.tensor(wav_chunk["wav"], device=current_device).unsqueeze(0).unsqueeze(0),
                size=int(len(wav_chunk["wav"]) / speed_ratio),
                mode='linear',
                align_corners=False
            ).squeeze()

            wav_chunk["wav"] = wav_resampled.cpu().numpy()

        output_path = os.path.join(output_dir, f"segment_{i:04d}_tts.wav")
        # Default sample rate 24000
        sample_rate = getattr(current_model.config, "sample_rate", 24000)

        try:
            torchaudio.save(output_path, torch.tensor(
                wav_chunk["wav"]).unsqueeze(0), sample_rate=sample_rate)
            output_files.append(output_path)
            print(f"[INFO] Saved TTS file: {output_path}")
        except Exception as e:
            print(f"[ERROR] Error saving file {output_path}: {e}")

# Clean up - free GPU memory
for model in models.values():
    del model
torch.cuda.empty_cache()

# Combine all audio files
combined_audio = AudioSegment.empty()
for file_path in output_files:
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        combined_audio += AudioSegment.from_file(file_path)
    else:
        print(f"[WARNING] File {file_path} is corrupted or empty and will be skipped.")

# Export final audio
final_output_path = os.path.join(output_dir, "final_tts_output.wav")
combined_audio.export(final_output_path, format="wav")
print(f"[INFO] Final TTS file saved: {final_output_path}")

# -----------------------------------------------
# 7. End of step Generate TTS translation
