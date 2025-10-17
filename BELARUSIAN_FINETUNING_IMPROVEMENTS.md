# XTTSv2 Belarusian Language Finetuning - Improvement Plan

**Date:** 2025-10-17
**Purpose:** Comprehensive analysis and improvement recommendations for Belarusian language finetuning

---

## Table of Contents
1. [Tokenizer Improvements](#1-tokenizer-improvements)
2. [Dataset Processing Enhancements](#2-dataset-processing-enhancements)
3. [Audio Format Analysis: WAV vs MP3](#3-audio-format-analysis-wav-vs-mp3)
4. [Finetuning & GPU Optimization](#4-finetuning--gpu-optimization)
5. [Best Practices Guidelines](#5-best-practices-guidelines)

---

## 1. Tokenizer Improvements

### Current Implementation Analysis
The current tokenizer implementation in `extend_vocab_config.py`:
- Uses simple BPE (Byte Pair Encoding) with Whitespace pre-tokenizer
- Extends vocabulary by merging old and new tokenizers
- Uses shell commands (`cat`, `tail`) for file operations (platform-dependent)

### Identified Issues
1. **Platform Dependency**: Uses Unix shell commands (`cat`, `tail`) which won't work on Windows
2. **No Text Normalization**: Missing Belarusian-specific text preprocessing
3. **Limited Pre-tokenization**: Only uses whitespace splitting
4. **No Validation**: Doesn't verify tokenization quality or coverage
5. **Hard-coded Vocab Size**: Default 2000 may not be optimal for Belarusian

### Recommended Improvements

#### A. **Add Belarusian Text Normalization**
```python
def normalize_belarusian_text(text: str) -> str:
    """
    Normalize Belarusian text before tokenization
    """
    # Convert to lowercase (optional, depends on use case)
    # text = text.lower()

    # Normalize Belarusian-specific characters
    # Handle ґ (U+0491) if needed

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Handle apostrophes consistently
    text = text.replace("'", "'")  # normalize apostrophe types

    return text
```

#### B. **Platform-Independent File Operations**
Replace shell commands with Python:
```python
def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # ... existing vocab merge code ...

    # Replace shell commands with Python
    import shutil

    # Copy old merges
    old_merges = os.path.join(old_tokenizer, 'merges.txt')
    new_merges = os.path.join(new_tokenizer, 'merges.txt')
    output_merges = os.path.join(save_dir, 'merges.txt')

    with open(output_merges, 'w', encoding='utf-8') as outfile:
        with open(old_merges, 'r', encoding='utf-8') as infile:
            outfile.write(infile.read())

        # Skip header from new merges
        with open(new_merges, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()[1:]  # Skip first line
            outfile.writelines(lines)
```

#### C. **Add Tokenization Quality Metrics**
```python
def analyze_tokenization_quality(tokenizer, texts, language):
    """
    Analyze tokenizer quality for the target language
    """
    total_chars = 0
    total_tokens = 0
    unk_count = 0

    for text in texts:
        tokens = tokenizer.encode(text, lang=language)
        total_chars += len(text)
        total_tokens += len(tokens)
        unk_count += tokens.count(1)  # UNK token ID

    compression_ratio = total_chars / total_tokens
    unk_ratio = unk_count / total_tokens

    print(f"Tokenization Analysis:")
    print(f"  Compression ratio: {compression_ratio:.2f} chars/token")
    print(f"  UNK token ratio: {unk_ratio*100:.2f}%")
    print(f"  Recommended: compression > 3.0, UNK < 1%")

    return compression_ratio, unk_ratio
```

#### D. **Optimize Vocabulary Size for Belarusian**
```python
# Recommended approach:
# 1. Start with 2000-3000 for small datasets (<10 hours)
# 2. Use 3000-5000 for medium datasets (10-50 hours)
# 3. Use 5000-8000 for large datasets (>50 hours)

# Add analysis to determine optimal size
def recommend_vocab_size(dataset_size_hours, num_unique_words):
    if dataset_size_hours < 10:
        base_size = 2000
    elif dataset_size_hours < 50:
        base_size = 3500
    else:
        base_size = 6000

    # Adjust based on vocabulary richness
    adjusted_size = min(base_size + (num_unique_words // 100), 8000)
    return adjusted_size
```

#### E. **Leverage Existing Belarusian Phonemizer**
The codebase already has Belarusian phonemizer support (`TTS/tts/utils/text/phonemizers/belarusian_phonemizer.py`):
- Integrates with fanetyka.jar for proper Belarusian phonemization
- Consider using phoneme-aware tokenization for better results
- Ensure `BEL_FANETYKA_JAR` environment variable is set

---

## 2. Dataset Processing Enhancements

### Current Implementation Analysis
Dataset classes in `TTS/tts/layers/xtts/trainer/`:
- `XTTSDataset`: Main dataset with text and audio
- `DVAEDataset`: Audio-only dataset for DVAE training
- Basic error handling with `failed_samples` tracking
- Random sampling by language

### Identified Issues
1. **No Data Augmentation**: Missing audio augmentation techniques
2. **Limited Validation**: Minimal quality checks on audio/text
3. **Inefficient Loading**: No caching or pre-loading mechanisms
4. **Hard-coded Thresholds**: Fixed minimum audio length (0.2s for XTTS, 0.5s for DVAE)
5. **No Speaker Diversity Tracking**: Doesn't monitor speaker distribution

### Recommended Improvements

#### A. **Audio Quality Validation**
```python
def validate_audio_quality(wav, sample_rate, filepath):
    """
    Comprehensive audio quality validation
    """
    issues = []

    # Check for clipping
    if torch.max(torch.abs(wav)) > 0.99:
        issues.append("clipping_detected")

    # Check for silence ratio
    silence_threshold = 0.01
    silence_ratio = (torch.abs(wav) < silence_threshold).float().mean()
    if silence_ratio > 0.5:
        issues.append("excessive_silence")

    # Check RMS energy
    rms = torch.sqrt(torch.mean(wav ** 2))
    if rms < 0.01:
        issues.append("low_energy")

    # Check for DC offset
    dc_offset = torch.mean(wav)
    if abs(dc_offset) > 0.1:
        issues.append("dc_offset")

    if issues:
        print(f"Quality issues in {filepath}: {', '.join(issues)}")
        return False
    return True
```

#### B. **Add Data Augmentation**
```python
class AudioAugmentation:
    """
    Audio augmentation for training robustness
    """
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def apply_random_gain(self, wav, min_gain=-3, max_gain=3):
        """Apply random gain in dB"""
        gain_db = random.uniform(min_gain, max_gain)
        gain_linear = 10 ** (gain_db / 20)
        return wav * gain_linear

    def add_gaussian_noise(self, wav, snr_db=40):
        """Add Gaussian noise at specified SNR"""
        signal_power = torch.mean(wav ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(wav) * torch.sqrt(noise_power)
        return wav + noise

    def apply_augmentation(self, wav, augment_prob=0.3):
        """Apply augmentation with probability"""
        if random.random() < augment_prob:
            wav = self.apply_random_gain(wav)
        if random.random() < augment_prob * 0.5:
            wav = self.add_gaussian_noise(wav)
        return wav.clip(-1, 1)
```

#### C. **Implement Audio Preprocessing Pipeline**
```python
def preprocess_belarusian_audio(wav, sample_rate=22050):
    """
    Belarusian-specific audio preprocessing
    """
    # 1. Normalize audio
    wav = wav / (torch.max(torch.abs(wav)) + 1e-8)
    wav = wav * 0.95  # Leave headroom

    # 2. Apply high-pass filter to remove DC offset and low-freq noise
    # (requires scipy or torchaudio filters)

    # 3. Apply subtle noise reduction if needed
    # (optional: use spectral subtraction or Wiener filtering)

    return wav
```

#### D. **Speaker-Aware Sampling**
```python
def get_speaker_balanced_batch(samples, batch_size):
    """
    Create batches with diverse speakers
    """
    # Group by speaker
    speaker_samples = {}
    for sample in samples:
        speaker = sample.get('speaker_name', 'unknown')
        if speaker not in speaker_samples:
            speaker_samples[speaker] = []
        speaker_samples[speaker].append(sample)

    # Sample from different speakers
    batch = []
    speakers = list(speaker_samples.keys())
    while len(batch) < batch_size:
        for speaker in speakers:
            if speaker_samples[speaker]:
                batch.append(speaker_samples[speaker].pop(0))
                if len(batch) >= batch_size:
                    break

    return batch
```

#### E. **Text Quality Validation**
```python
def validate_belarusian_text(text, min_length=3, max_length=500):
    """
    Validate Belarusian text quality
    """
    # Check length
    if len(text) < min_length or len(text) > max_length:
        return False, "invalid_length"

    # Check for valid Belarusian characters
    # Belarusian Cyrillic: а-я, ё, і, ў, ', -
    valid_chars_pattern = r'^[а-яёіўА-ЯЁІЎ\s\'\-\,\.\!\?\:\;\"0-9]+$'
    import re
    if not re.match(valid_chars_pattern, text):
        return False, "invalid_characters"

    # Check for reasonable word count
    words = text.split()
    if len(words) < 1:
        return False, "no_words"

    # Check for excessive repetition
    if len(set(words)) < len(words) * 0.3:
        return False, "excessive_repetition"

    return True, "valid"
```

---

## 3. Audio Format Analysis: WAV vs MP3

### Technical Comparison

#### WAV Format
**Advantages:**
- ✅ Lossless quality - no compression artifacts
- ✅ Faster to load (no decompression needed)
- ✅ More predictable behavior
- ✅ No licensing concerns
- ✅ Better for training (preserves all spectral information)

**Disadvantages:**
- ❌ Large file size (~10x larger than MP3)
- ❌ Higher storage and bandwidth requirements
- ❌ Slower disk I/O for large datasets

**Recommended Specifications:**
- Sample rate: 22050 Hz (XTTS v2 default)
- Bit depth: 16-bit (sufficient for TTS)
- Channels: Mono
- File size: ~2.6 MB per minute

#### MP3 Format
**Advantages:**
- ✅ Small file size (~10x smaller)
- ✅ Faster data transfer
- ✅ CommonVoice dataset uses MP3 (32kHz, 48kbps)

**Disadvantages:**
- ❌ Lossy compression - removes high frequencies
- ❌ Compression artifacts (especially at low bitrates)
- ❌ Slower to load (decompression overhead)
- ❌ Quality loss in training data
- ❌ Potential issues with multiple encode/decode cycles

### Belarusian CommonVoice Dataset Analysis
Based on `recipes/bel-alex73/README.md`:
- Format: MP3, 32kHz, 48kbps
- Quality: Sufficient but not optimal
- Recommendation: "Better voice synthesis requires specific corpus with good pronunciation and good record quality"

### Recommendations

#### **For Production/Best Results:**
```
Use WAV format exclusively:
- 22050 Hz sample rate (XTTS v2 native)
- 16-bit depth
- Mono channel
- Store in uncompressed PCM format
```

#### **For Prototyping/Limited Storage:**
```
Use MP3 with high quality settings:
- Minimum 128 kbps (preferably 192-320 kbps)
- 22050 Hz or 44100 Hz
- Convert to WAV during preprocessing (done once)
```

#### **Processing Pipeline:**
```python
def convert_dataset_to_optimal_format(input_dir, output_dir):
    """
    Convert any audio format to optimal WAV for training
    """
    import torchaudio
    import os
    from pathlib import Path

    target_sr = 22050

    for audio_file in Path(input_dir).rglob('*.*'):
        if audio_file.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']:
            # Load audio
            wav, sr = torchaudio.load(str(audio_file))

            # Convert to mono
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample to target rate
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)

            # Normalize
            wav = wav / (torch.max(torch.abs(wav)) + 1e-8) * 0.95

            # Save as WAV
            output_path = Path(output_dir) / audio_file.relative_to(input_dir).with_suffix('.wav')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), wav, target_sr, bits_per_sample=16)
```

### Performance Impact Analysis

| Metric | MP3 (48kbps) | MP3 (192kbps) | WAV (16-bit) |
|--------|--------------|---------------|--------------|
| File Size (1 min) | ~0.36 MB | ~1.4 MB | ~2.6 MB |
| Load Time | ~15ms | ~15ms | ~5ms |
| Quality Loss | High | Minimal | None |
| Training Impact | Moderate | Low | None |
| Recommended | ❌ No | ⚠️ Acceptable | ✅ Yes |

### **Final Recommendation:**
**Convert all MP3 to WAV during dataset preparation** (one-time cost), then train on WAV files. The quality improvement and faster loading outweigh the storage cost.

---

## 4. Finetuning & GPU Optimization

### Current Implementation Analysis

#### Existing Optimizations:
- ✅ TF32 support (optional via command-line flags)
- ✅ Gradient accumulation
- ✅ Multi-dataset support
- ✅ Checkpoint saving strategy

#### Missing Optimizations:
- ❌ No mixed precision training (FP16/BF16)
- ❌ No gradient checkpointing
- ❌ DataLoader not optimized (num_workers, pin_memory)
- ❌ No model compilation (torch.compile)
- ❌ No DeepSpeed integration for multi-GPU
- ❌ No gradient clipping monitoring

### Recommended Improvements

#### A. **Add Mixed Precision Training**
```python
# Add to train_gpt_xtts.py

import argparse

parser.add_argument("--use_amp", action="store_true",
                   help="Use Automatic Mixed Precision (FP16)")
parser.add_argument("--use_bfloat16", action="store_true",
                   help="Use BFloat16 instead of FP16 (better for Ampere+ GPUs)")

# In training config:
from torch.cuda.amp import autocast, GradScaler

if args.use_amp or args.use_bfloat16:
    dtype = torch.bfloat16 if args.use_bfloat16 else torch.float16
    scaler = GradScaler() if args.use_amp and not args.use_bfloat16 else None

    # Training loop modifications needed in GPTTrainer
    # Use autocast context:
    # with autocast(dtype=dtype):
    #     loss = model(...)
```

#### B. **Optimize DataLoader Configuration**
```python
# Add to GPTTrainerConfig or train_gpt_xtts.py

config.num_loader_workers = 8  # Increase from current 4
config.pin_memory = True  # Enable for faster GPU transfer
config.persistent_workers = True  # Keep workers alive between epochs
config.prefetch_factor = 2  # Prefetch batches

# For multi-GPU setups
config.dataloader_num_workers = min(8, os.cpu_count() // torch.cuda.device_count())
```

#### C. **Add Gradient Checkpointing**
```python
# In GPTTrainer initialization

def enable_gradient_checkpointing(self):
    """
    Enable gradient checkpointing to reduce memory usage
    Trades compute for memory - useful for large models
    """
    if hasattr(self.xtts.gpt, 'gradient_checkpointing_enable'):
        self.xtts.gpt.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
```

#### D. **Implement Torch 2.0 Compilation**
```python
# Add to train_gpt_xtts.py

parser.add_argument("--compile_model", action="store_true",
                   help="Use torch.compile for faster training (PyTorch 2.0+)")

# After model initialization
if args.compile_model and hasattr(torch, 'compile'):
    model.xtts.gpt = torch.compile(
        model.xtts.gpt,
        mode="reduce-overhead",  # or "max-autotune" for best performance
        fullgraph=False
    )
    print("✓ Model compiled with torch.compile")
```

#### E. **Enhanced GPU Memory Optimization**
```python
def optimize_gpu_memory():
    """
    Apply GPU memory optimizations
    """
    # Enable TF32 for Ampere+ GPUs (faster matmul)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Set memory allocator settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
```

#### F. **Add Gradient Monitoring**
```python
def log_gradient_stats(model, step):
    """
    Monitor gradient statistics to detect issues
    """
    total_norm = 0.0
    num_params = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1

    total_norm = total_norm ** 0.5

    # Log to tensorboard
    if step % 100 == 0:
        print(f"Step {step} - Grad Norm: {total_norm:.4f}, Params with grad: {num_params}")

    # Warning for gradient explosion
    if total_norm > 100:
        print(f"⚠️  WARNING: Large gradient norm detected: {total_norm:.2f}")
```

#### G. **Smart Batch Size Selection**
```python
def find_optimal_batch_size(model, sample_batch, max_batch_size=32):
    """
    Find optimal batch size through binary search
    """
    def test_batch_size(bs):
        try:
            # Test forward pass
            torch.cuda.empty_cache()
            with torch.no_grad():
                model(**sample_batch)
            return True
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e

    low, high = 1, max_batch_size
    optimal_bs = 1

    while low <= high:
        mid = (low + high) // 2
        if test_batch_size(mid):
            optimal_bs = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"✓ Optimal batch size: {optimal_bs}")
    return optimal_bs
```

#### H. **Multi-GPU DeepSpeed Integration**
```python
# Add DeepSpeed config for distributed training

DEEPSPEED_CONFIG = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-6,
            "betas": [0.9, 0.96],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,  # Stage 2 for memory efficiency
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "reduce_scatter": True
    }
}
```

### Performance Optimization Checklist

- [ ] Enable TF32 on Ampere+ GPUs (3090, 4090, A100, etc.)
- [ ] Use mixed precision (BFloat16 for Ampere+, FP16 for older GPUs)
- [ ] Set num_workers = 8-12 for DataLoader
- [ ] Enable pin_memory and persistent_workers
- [ ] Use gradient accumulation for larger effective batch size
- [ ] Apply gradient checkpointing if OOM
- [ ] Use torch.compile on PyTorch 2.0+
- [ ] Monitor gradient norms for stability
- [ ] Profile with torch.profiler to find bottlenecks
- [ ] Consider DeepSpeed for multi-GPU training

---

## 5. Best Practices Guidelines

### Dataset Preparation

#### **1. Data Collection**
```
Recommended dataset characteristics for Belarusian:
- Minimum: 10-20 hours of high-quality audio
- Optimal: 30-50 hours
- Sample rate: 22050 Hz
- Format: WAV (16-bit, mono)
- Speakers: 3-10 diverse speakers (avoid single speaker)
- Recording quality: Studio or high-quality environment
- Text quality: Proper Belarusian orthography
```

#### **2. Data Cleaning Pipeline**
```bash
# Step 1: Convert all audio to WAV
python scripts/convert_to_wav.py --input datasets/raw --output datasets/wav

# Step 2: Validate audio quality
python scripts/validate_audio.py --input datasets/wav --min_duration 1.0 --max_duration 15.0

# Step 3: Normalize text
python scripts/normalize_belarusian_text.py --input metadata_raw.csv --output metadata_clean.csv

# Step 4: Split train/eval
python scripts/split_dataset.py --input metadata_clean.csv --train_ratio 0.95
```

#### **3. Metadata Format**
```csv
audio_file|text|speaker_name
wavs/speaker1_001.wav|Добры дзень, як справы?|speaker1
wavs/speaker1_002.wav|Дзякуй, усё добра.|speaker1
wavs/speaker2_001.wav|Сёння выдатная надвор'е.|speaker2
```

### Training Configuration

#### **Optimal Hyperparameters for Belarusian**

```python
# For small datasets (10-20 hours)
CONFIG_SMALL = {
    "num_epochs": 10,
    "batch_size": 4,
    "grad_acumm": 8,  # effective batch size = 32
    "max_text_length": 300,
    "max_audio_length": 255995,
    "lr": 5e-6,
    "extended_vocab_size": 2500,
    "save_step": 5000,
    "weight_decay": 1e-2,
}

# For medium datasets (20-50 hours)
CONFIG_MEDIUM = {
    "num_epochs": 8,
    "batch_size": 8,
    "grad_acumm": 4,  # effective batch size = 32
    "max_text_length": 400,
    "max_audio_length": 330750,
    "lr": 5e-6,
    "extended_vocab_size": 4000,
    "save_step": 10000,
    "weight_decay": 1e-2,
}

# For large datasets (>50 hours)
CONFIG_LARGE = {
    "num_epochs": 5,
    "batch_size": 12,
    "grad_acumm": 3,  # effective batch size = 36
    "max_text_length": 500,
    "max_audio_length": 440500,
    "lr": 3e-6,  # lower LR for stability
    "extended_vocab_size": 6000,
    "save_step": 15000,
    "weight_decay": 1e-2,
}
```

#### **GPU-Specific Recommendations**

```python
# RTX 3090 / 4090 (24GB VRAM)
GPU_CONFIG_24GB = {
    "batch_size": 8,
    "use_bfloat16": True,  # Native support on Ampere+
    "num_workers": 8,
    "tf32_matmul": True,
    "tf32_cudnn": True,
    "compile_model": True,
}

# RTX 4080 / 3080 Ti (16GB VRAM)
GPU_CONFIG_16GB = {
    "batch_size": 4,
    "use_amp": True,  # FP16
    "gradient_checkpointing": True,
    "num_workers": 6,
    "tf32_matmul": True,
    "tf32_cudnn": True,
}

# RTX 3060 / 4060 Ti (12GB VRAM)
GPU_CONFIG_12GB = {
    "batch_size": 2,
    "grad_acumm": 8,
    "use_amp": True,
    "gradient_checkpointing": True,
    "num_workers": 4,
    "max_audio_length": 220000,  # Reduce to save memory
}
```

### Training Pipeline

#### **Complete Training Workflow**

```bash
# 1. Setup environment
export BEL_FANETYKA_JAR=/path/to/fanetyka.jar

# 2. Download pretrained checkpoint
python download_checkpoint.py --output_path checkpoints/

# 3. Extend vocabulary for Belarusian
python extend_vocab_config.py \
  --output_path=checkpoints/ \
  --metadata_path=datasets/metadata_train.csv \
  --language=be \
  --extended_vocab_size=4000

# 4. OPTIONAL: Train DVAE (if dataset < 20 hours)
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
  --output_path=checkpoints/ \
  --train_csv_path=datasets/metadata_train.csv \
  --eval_csv_path=datasets/metadata_eval.csv \
  --language=be \
  --num_epochs=5 \
  --batch_size=512 \
  --lr=5e-6

# 5. Train GPT (main step)
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path=checkpoints/ \
  --metadatas=datasets/metadata_train.csv,datasets/metadata_eval.csv,be \
  --num_epochs=8 \
  --batch_size=8 \
  --grad_acumm=4 \
  --max_text_length=400 \
  --max_audio_length=330750 \
  --weight_decay=1e-2 \
  --lr=5e-6 \
  --save_step=10000 \
  --tf32_matmul=True \
  --tf32_cudnn=True

# 6. Monitor training
tensorboard --logdir checkpoints/

# 7. Evaluate checkpoints
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/GPT_XTTS_FT-*/best_model.pth \
  --test_texts test_sentences_be.txt
```

### Monitoring & Validation

#### **Key Metrics to Track**

1. **Training Loss**
   - Target: Steady decrease, should reach < 2.0
   - Warning: If loss plateaus above 3.0, check data quality

2. **Alignment Attention**
   - Check attention plots in TensorBoard
   - Should show clear diagonal patterns

3. **Audio Quality** (via TensorBoard)
   - Listen to evaluation samples every 5000 steps
   - Check for: clarity, naturalness, prosody

4. **Text Coverage**
   - Verify UNK token rate < 1%
   - Check tokenization quality regularly

5. **GPU Utilization**
   ```bash
   nvidia-smi dmon -s u -d 5  # Monitor GPU usage
   # Target: >90% utilization
   ```

### Common Issues & Solutions

#### **Issue 1: Out of Memory (OOM)**
```
Solutions:
1. Reduce batch_size by 50%
2. Increase grad_acumm proportionally
3. Reduce max_audio_length by 20%
4. Enable gradient_checkpointing
5. Use mixed precision training
```

#### **Issue 2: Poor Audio Quality**
```
Root Causes:
- Low-quality training data (MP3 artifacts)
- Insufficient training (< 50k steps)
- Wrong sample rate conversion
- Clipping in audio files

Solutions:
1. Re-validate dataset audio quality
2. Convert all to WAV with proper normalization
3. Train for more steps
4. Check for data augmentation issues
```

#### **Issue 3: Language Mixing / Code-Switching**
```
Root Causes:
- Belarusian text contains Russian/English words
- Tokenizer not properly extended
- Dataset has mixed languages

Solutions:
1. Clean text to pure Belarusian
2. Increase extended_vocab_size
3. Verify language field in metadata
4. Use phonemizer for better coverage
```

#### **Issue 4: Slow Convergence**
```
Solutions:
1. Verify learning rate (5e-6 recommended)
2. Increase effective batch size (batch_size × grad_acumm)
3. Check gradient norms (should be 0.1-10.0)
4. Ensure TF32 is enabled on compatible GPUs
5. Profile code to find bottlenecks
```

### Final Recommendations Summary

**Priority 1 (Critical):**
1. ✅ Convert all audio to WAV format (22050 Hz, 16-bit, mono)
2. ✅ Implement platform-independent tokenizer (remove shell commands)
3. ✅ Add audio quality validation before training
4. ✅ Enable TF32 on Ampere+ GPUs
5. ✅ Optimize DataLoader (num_workers, pin_memory)

**Priority 2 (Important):**
6. ✅ Add mixed precision training (BFloat16 for modern GPUs)
7. ✅ Implement data augmentation (gain, noise)
8. ✅ Add gradient monitoring and clipping
9. ✅ Use torch.compile for PyTorch 2.0+
10. ✅ Add tokenization quality metrics

**Priority 3 (Nice to Have):**
11. ✅ Implement gradient checkpointing for memory efficiency
12. ✅ Add speaker-aware batch sampling
13. ✅ Create automated dataset validation pipeline
14. ✅ Set up comprehensive monitoring dashboard
15. ✅ Implement DeepSpeed for multi-GPU setups

---

## Appendix: Belarusian-Specific Notes

### Belarusian Language Characteristics
- **Alphabet**: Cyrillic with specific letters: і, ў, apostrophe (')
- **Phonetics**: Different from Russian, requires proper phonemizer
- **Resources**: fanetyka.jar (included in recipes/bel-alex73)
- **Datasets**: CommonVoice (90 hours), but quality-over-quantity applies

### Environment Setup for Belarusian
```bash
# Install required packages
pip install jpype1  # For fanetyka.jar

# Download fanetyka.jar
wget https://github.com/alex73/Software-Korpus/releases/latest/download/fanetyka.jar

# Set environment variable
export BEL_FANETYKA_JAR=/path/to/fanetyka.jar

# Verify phonemizer works
python -c "from TTS.tts.utils.text.belarusian.phonemizer import belarusian_text_to_phonemes; print(belarusian_text_to_phonemes('Добры дзень'))"
```

### Reference Implementation
The codebase includes Belarusian training example at `recipes/bel-alex73/`:
- Uses GlowTTS + HiFiGAN architecture (older approach)
- Recommends 30 hours of quality data over 90 hours of mixed quality
- Trains for ~24k steps on 24GB GPU
- For XTTS v2, adapt similar principles but with modern architecture

---

**Document Version:** 1.0
**Last Updated:** 2025-10-17
**Author:** Claude Code Analysis
**Status:** Ready for Implementation
