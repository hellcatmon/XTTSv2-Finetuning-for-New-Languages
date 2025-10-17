# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a finetuning framework for XTTSv2 (Text-to-Speech model) adapted for new languages, with Vietnamese and Belarusian as primary examples. The project enables transfer learning on XTTS models to support languages not originally included in the base model.

## 🆕 Recent Improvements (P1)

**Priority 1 improvements have been implemented.** See `P1_IMPROVEMENTS_README.md` for quick start.

### New Modules:
1. **`convert_audio_to_wav.py`** - Platform-independent audio conversion (MP3→WAV)
2. **`extend_vocab_config_improved.py`** - Enhanced tokenizer with text normalization
3. **`validate_audio_dataset.py`** - Comprehensive audio quality validation

### Optimizations:
- TF32 support for Ampere+ GPUs (3090, 4090, A100)
- DataLoader optimization (8 workers, pin_memory, persistent_workers)
- GPU memory optimization and benchmarking

## Architecture

### Core Training Pipeline

The finetuning process follows a sequential pipeline:

1. **Checkpoint Download** (`download_checkpoint.py`)
   - Downloads pretrained XTTSv2 model files from Coqui servers
   - Retrieves DVAE checkpoint, mel stats, vocab.json, model.pth, and config.json
   - Default output: `checkpoints/XTTS_v2.0_original_model_files/`

2. **Vocabulary Extension** (`extend_vocab_config.py`)
   - Extends the tokenizer with new language-specific vocabulary
   - Trains new BPE tokenizer on target language text
   - Merges original and new tokenizers
   - Updates config.json to include new language
   - Critical for proper text encoding in target language

3. **DVAE Finetuning** (`train_dvae_xtts.py`) - Optional
   - Finetunes the Discrete Variational Autoencoder on target language audio
   - Only needed if dataset has limited short texts (~20 hours)
   - Uses mel spectrogram features with proper padding handling (divisible by 4)
   - Saves best checkpoint based on evaluation loss

4. **GPT Finetuning** (`train_gpt_xtts.py`)
   - Main finetuning step for the GPT-based acoustic model
   - Supports multi-dataset training via `--metadatas` parameter
   - Uses gradient accumulation for large effective batch sizes
   - Implements learning rate scheduling with MultiStepLR
   - Critical hyperparameters: max_text_length, max_audio_length

### TTS Library Structure

The `TTS/` directory contains the complete Coqui TTS library with:
- **tts/layers/xtts/**: XTTS model architecture and training components
- **tts/configs/**: Configuration dataclasses for all model types
- **tts/datasets/**: Dataset loaders and formatters
- **recipes/**: Training recipes for various datasets and models (ljspeech, vctk, multilingual, etc.)

### Projects Directory

Contains application-level implementations:
- **video_translator.py**: Full pipeline for video translation with TTS
- **video_translator_speedup.py**: Multi-GPU optimized version using round-robin device assignment
- **video_translator_gpus.py**: Alternative GPU utilization implementation
- **combine_all_steps.py**: Pipeline integration script

## Common Commands

### Environment Setup

**Fast Installation with uv (Recommended - 10-100x faster than pip):**
```bash
./setup_with_uv.sh
```

Traditional pip installation:
```bash
pip install -r requirements.txt
```

### Download Pretrained Model
```bash
python download_checkpoint.py --output_path checkpoints/
```

### Extend Vocabulary (Required for new languages)
```bash
python extend_vocab_config.py \
  --output_path=checkpoints/ \
  --metadata_path=datasets/metadata_train.csv \
  --language=vi \
  --extended_vocab_size=2000
```

### Train DVAE (Optional - skip if >20 hours of data)
```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
  --output_path=checkpoints/ \
  --train_csv_path=datasets/metadata_train.csv \
  --eval_csv_path=datasets/metadata_eval.csv \
  --language=vi \
  --num_epochs=5 \
  --batch_size=512 \
  --lr=5e-6
```

### Train GPT (Main finetuning step)

Single dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path=checkpoints/ \
  --metadatas=datasets/metadata_train.csv,datasets/metadata_eval.csv,vi \
  --num_epochs=5 \
  --batch_size=8 \
  --grad_acumm=4 \
  --max_text_length=400 \
  --max_audio_length=330750 \
  --weight_decay=1e-2 \
  --lr=5e-6 \
  --save_step=50000
```

Multiple datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path=checkpoints/ \
  --metadatas \
    datasets-1/metadata_train.csv,datasets-1/metadata_eval.csv,vi \
    datasets-2/metadata_train.csv,datasets-2/metadata_eval.csv,vi \
  --num_epochs=5 \
  --batch_size=8 \
  --grad_acumm=4 \
  --max_text_length=400 \
  --max_audio_length=330750 \
  --weight_decay=1e-2 \
  --lr=5e-6 \
  --save_step=50000
```

### Shell Scripts
- `train_dvae_xtts.sh`: Convenience script for DVAE training
- `train_gpt_xtts.sh`: Convenience script for GPT training

## Dataset Format

Metadata CSV files must follow this format:
```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|@X
wavs/yyy.wav|Nice to meet you.|@Y
wavs/zzz.wav|Good to see you.|@Z
```

Directory structure:
```
datasets/
├── wavs/
│   ├── xxx.wav
│   ├── yyy.wav
│   └── ...
├── metadata_train.csv
└── metadata_eval.csv
```

## Key Configuration Details

### GPT Training Parameters
- **max_audio_length**: Maximum audio length in samples (~11.6 seconds at 255995)
- **max_text_length**: Maximum text length in characters
- **grad_acumm**: Gradient accumulation steps for effective larger batch size
- **save_step**: Checkpoint saving frequency
- **lr_scheduler**: Uses MultiStepLR with milestones at [save_step*3, save_step*6, save_step*9]

### Model Architecture
- GPT uses perceiver resampler and masking for ground truth prompt
- Audio tokens: 1024 (start), 1025 (stop), 1026 total tokens
- Conditioning: 12 seconds max, 4 seconds min
- Sample rates: 22050 (training), 24000 (output)

### TF32 Optimization
Training scripts support TF32 acceleration:
- `--tf32_matmul`: Enable TF32 for matrix multiplication
- `--tf32_cudnn`: Enable TF32 for cuDNN operations

## Inference

Load finetuned model:
```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("checkpoints/GPT_XTTS_FT-{timestamp}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path="checkpoints/GPT_XTTS_FT-{timestamp}/best_model.pth",
    vocab_path="checkpoints/XTTS_v2.0_original_model_files/vocab.json",
    use_deepspeed=False
)
model.to(device)
```

Generate speech:
```python
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=model.config.gpt_cond_len,
    max_ref_length=model.config.max_ref_len,
    sound_norm_refs=model.config.sound_norm_refs,
)

wav_chunk = model.inference(
    text=text,
    language="vi",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.1,
    length_penalty=1.0,
    repetition_penalty=10.0,
    top_k=10,
    top_p=0.3,
)
```

## Important Notes

- **HiFiGAN finetuning**: Not recommended - degrades performance
- **DVAE finetuning**: Only needed for datasets with <20 hours of short texts
- **Tokenizer extension**: Always required before training on new language
- **Multi-GPU**: Projects contain examples of multi-GPU inference using round-robin device assignment
- **Vietnamese model**: Pre-trained Vietnamese model available at [anhnh2002/vnTTS](https://huggingface.co/anhnh2002/vnTTS)
