# Quick Reference Card - P1 Improvements

## 🚀 One-Line Commands

### 1. Convert Audio (MP3 → WAV)
```bash
python convert_audio_to_wav.py --input_dir raw --output_dir wav --update_metadata meta.csv --output_metadata meta_wav.csv
```

### 2. Validate Audio Quality
```bash
python validate_audio_dataset.py --metadata meta_wav.csv --fix_issues --output_fixed meta_clean.csv
```

### 3. Extend Vocabulary
```bash
python extend_vocab_config_improved.py --output_path checkpoints/ --metadata_path meta_train.csv --language be --extended_vocab_size 4000
```

### 4. Train (All Optimizations)
```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py --output_path checkpoints/ --metadatas meta_train.csv,meta_eval.csv,be --batch_size 8 --grad_acumm 4 --tf32_matmul=True --tf32_cudnn=True
```

---

## 📋 5-Step Pipeline

```bash
# Step 1: Convert
python convert_audio_to_wav.py --input_dir raw --output_dir wav --update_metadata meta.csv --output_metadata meta_wav.csv

# Step 2: Validate
python validate_audio_dataset.py --metadata meta_wav.csv --fix_issues --output_fixed meta_clean.csv

# Step 3: Split (95/5)
python -c "import pandas as pd; df=pd.read_csv('meta_clean.csv',sep='|',header=None).sample(frac=1,random_state=42); s=int(len(df)*0.95); df[:s].to_csv('meta_train.csv',sep='|',header=False,index=False); df[s:].to_csv('meta_eval.csv',sep='|',header=False,index=False)"

# Step 4: Extend Vocab
python extend_vocab_config_improved.py --output_path checkpoints/ --metadata_path meta_train.csv --language be --extended_vocab_size 4000

# Step 5: Train
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py --output_path checkpoints/ --metadatas meta_train.csv,meta_eval.csv,be --num_epochs 8 --batch_size 8 --grad_acumm 4 --tf32_matmul=True --tf32_cudnn=True
```

---

## 🎯 GPU Settings

### RTX 4090 / A100
```bash
--batch_size=8 --grad_acumm=4 --tf32_matmul=True --tf32_cudnn=True
```

### RTX 3090 / 3080 Ti
```bash
--batch_size=6 --grad_acumm=6 --tf32_matmul=True --tf32_cudnn=True
```

### RTX 3060 / 2080 Ti
```bash
--batch_size=4 --grad_acumm=8
```

---

## 📊 Key Parameters

### Audio Conversion
| Param | Default | Description |
|-------|---------|-------------|
| `--sample_rate` | 22050 | Target Hz |
| `--bit_depth` | 16 | Bit depth |
| `--normalize_peak` | 0.95 | Peak level |

### Validation
| Param | Default | Description |
|-------|---------|-------------|
| `--min_duration` | 1.0 | Min seconds |
| `--max_duration` | 15.0 | Max seconds |
| `--fix_issues` | False | Auto cleanup |

### Tokenizer
| Param | Default | Description |
|-------|---------|-------------|
| `--extended_vocab_size` | 4000 | Vocab size |
| `--auto_vocab_size` | False | Auto recommend |
| `--language` | Required | Lang code |

### Training
| Param | Default | Description |
|-------|---------|-------------|
| `--batch_size` | 8 | Batch size |
| `--grad_acumm` | 1 | Grad steps |
| `--tf32_matmul` | False | TF32 enable |
| `--lr` | 5e-6 | Learning rate |

---

## 🔧 Troubleshooting

### OOM Error
```bash
--batch_size=4 --grad_acumm=8
```

### Poor Audio Quality
```bash
# Re-validate and clean
python validate_audio_dataset.py --metadata meta.csv --fix_issues --output_fixed meta_clean.csv
```

### Low Tokenization Quality
```bash
# Increase vocab or auto-size
--extended_vocab_size=6000
# OR
--auto_vocab_size --dataset_hours=30
```

---

## 📈 Performance Gains

| Optimization | Speedup |
|--------------|---------|
| TF32 (4090) | ~1.8x |
| TF32 (3090) | ~1.6x |
| DataLoader | ~2-3x |
| Combined | ~3-5x |

---

## 📁 New Files

**Modules:**
- `convert_audio_to_wav.py`
- `extend_vocab_config_improved.py`
- `validate_audio_dataset.py`

**Docs:**
- `P1_IMPROVEMENTS_README.md` (Quick Start)
- `P1_IMPROVEMENTS_GUIDE.md` (Full Guide)
- `IMPLEMENTATION_SUMMARY.md` (Summary)

---

## ✅ Validation Checks

1. ✓ Sample Rate (22050 Hz)
2. ✓ Duration (1-15 sec)
3. ✓ Clipping (< 0.99)
4. ✓ Silence (< 50%)
5. ✓ Energy (> 0.01)
6. ✓ DC Offset (< 0.1)

---

## 📞 Help

- Quick Start: `P1_IMPROVEMENTS_README.md`
- Full Guide: `P1_IMPROVEMENTS_GUIDE.md`
- Analysis: `BELARUSIAN_FINETUNING_IMPROVEMENTS.md`
