# XTTSv2 Setup Guide - Python 3.11 Environment

This guide provides comprehensive instructions for setting up the XTTSv2 finetuning environment with Python 3.11.

## 🚀 Quick Start

### Option 1: Standalone Setup (Recommended for new servers)

Run this from **any directory** to set up everything from scratch:

```bash
# Download the script
curl -O https://raw.githubusercontent.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages/feature/improvements/standalone_setup.sh

# Make it executable
chmod +x standalone_setup.sh

# Run the setup
./standalone_setup.sh
```

### Option 2: Setup in Existing Project

If you already have the project cloned:

```bash
cd XTTSv2-Finetuning-for-New-Languages
./setup_env_py311.sh
```

## 📋 What the Scripts Do

### Step-by-Step Process

1. **Python 3.11 Installation** ✅
   - Checks if Python 3.11 is installed
   - On Linux: Automatically installs via deadsnakes PPA
   - Verifies installation and version

2. **UV Package Manager** ✅
   - Installs `uv` (10-100x faster than pip)
   - Falls back to pip if uv fails

3. **Git Clone** ✅ (standalone_setup.sh only)
   - Clones: `https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git`
   - Branch: `feature/improvements`
   - Handles existing directories intelligently

4. **Virtual Environment** ✅
   - Creates Python 3.11 virtual environment
   - Activates the venv
   - Verifies Python 3.11 is being used

5. **Dependencies Installation** ✅
   - Upgrades pip
   - Installs all requirements with UV (explicitly for Python 3.11)
   - Verifies packages are installed for Python 3.11

6. **Jupyter Kernel Registration** ✅
   - Installs ipykernel
   - Registers Python 3.11 kernel as "python311_xtts"
   - Enables use in Jupyter notebooks

7. **Verification** ✅
   - Checks critical files exist
   - Verifies CUDA availability
   - Confirms python3.11 command is available

8. **Configuration Summary** ✅
   - Shows complete environment configuration
   - Displays hardware and CUDA info
   - Provides next steps

## 🔑 Key Features

### Python 3.11 Guarantee

The scripts ensure:

- ✅ Python 3.11 is installed and available as `python3.11` command
- ✅ Virtual environment uses Python 3.11
- ✅ All packages are installed specifically for Python 3.11
- ✅ Jupyter kernel is registered for Python 3.11

### UV Package Manager

- 📦 **10-100x faster** than pip
- 📦 Explicit Python interpreter targeting (`--python $(which python)`)
- 📦 Automatic fallback to pip if needed

### Jupyter Notebook Support

After setup, you can:

1. Start Jupyter:
   ```bash
   cd XTTSv2-Finetuning-for-New-Languages
   source venv/bin/activate
   jupyter notebook
   ```

2. Select kernel: **"Python 3.11 (XTTSv2)"**

3. Or use `python3.11` command directly in cells:
   ```python
   !python3.11 --version
   ```

## 📊 Configuration Summary Example

After successful setup, you'll see:

```
🐍 Python Environment:
  Python Version:        3.11.x
  Python Path:           /path/to/venv/bin/python
  python3.11 Command:    /usr/bin/python3.11
  Virtual Environment:   venv
  Pip Version:           24.x
  Jupyter Kernel:        python311_xtts (Python 3.11 (XTTSv2))

📦 Package Management:
  uv Version:            0.x.x
  Installation Method:   uv
  Packages Installed:    XXX

📁 Project Information:
  Repository:            https://github.com/hellcatmon/...
  Branch:                feature/improvements
  Commit:                xxxxxxx
  Install Directory:     /path/to/XTTSv2-Finetuning-for-New-Languages

🖥️  Hardware & CUDA:
  Operating System:      linux-gnu
  CUDA Available:        Yes
  CUDA Version:          12.x
  GPU Count:             1
  GPU Model:             NVIDIA RTX 4090

⏱️  Installation Time:
  Total Time:            5m 32s
```

## 🎯 Using Python 3.11

### In Terminal

```bash
# Activate environment
source venv/bin/activate

# Python in venv is 3.11
python --version  # Python 3.11.x

# Or use python3.11 directly
python3.11 --version
```

### In Jupyter Notebooks

```python
# Method 1: Select "Python 3.11 (XTTSv2)" kernel from the kernel menu

# Method 2: Use python3.11 command in cells
!python3.11 train_gpt_xtts.py --help

# Method 3: Verify in notebook
import sys
print(f"Python version: {sys.version}")  # Should show 3.11.x
```

### In Scripts

```bash
# Training script
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
  --output_path=checkpoints/ \
  --metadatas=datasets/metadata_train.csv,datasets/metadata_eval.csv,vi \
  --num_epochs=5 --batch_size=8
```

## 🔧 Troubleshooting

### Python 3.11 not found

If `python3.11` command is not available after setup:

**On Linux:**
```bash
sudo apt-get update
sudo apt-get install python3.11
```

**On macOS:**
```bash
brew install python@3.11
```

### Packages installed for wrong Python version

Re-run the setup script:
```bash
./standalone_setup.sh
# Or
./setup_env_py311.sh
```

The script will verify and reinstall if needed.

### Jupyter kernel not showing

```bash
# Activate venv
source venv/bin/activate

# Reinstall kernel
python -m ipykernel install --user --name=python311_xtts --display-name="Python 3.11 (XTTSv2)"

# List available kernels
jupyter kernelspec list
```

### UV installation failed

The scripts automatically fall back to pip. You can manually install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

## 📚 Next Steps After Setup

1. **Download pretrained model:**
   ```bash
   python download_checkpoint.py --output_path checkpoints/
   ```

2. **Extend vocabulary for new language:**
   ```bash
   python extend_vocab_config_improved.py \
     --output_path=checkpoints/ \
     --metadata_path=datasets/metadata_train.csv \
     --language=vi
   ```

3. **Train GPT model:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
     --output_path=checkpoints/ \
     --metadatas=datasets/metadata_train.csv,datasets/metadata_eval.csv,vi \
     --num_epochs=5 --batch_size=8
   ```

## 📖 Additional Resources

- **CLAUDE.md** - Project overview and detailed instructions
- **P1_IMPROVEMENTS_README.md** - Recent improvements (P1)
- **README.md** - Original README

## ✅ Verification Checklist

After running the setup script, verify:

- [ ] `python3.11 --version` shows Python 3.11.x
- [ ] `python --version` (in venv) shows Python 3.11.x
- [ ] `pip list | grep torch` shows PyTorch installed
- [ ] `jupyter kernelspec list` shows `python311_xtts`
- [ ] All training scripts are present in the directory
- [ ] CUDA is available (if using GPU): `python -c "import torch; print(torch.cuda.is_available())"`

## 🆘 Support

If you encounter issues:

1. Check the script output for error messages
2. Verify Python 3.11 installation: `which python3.11`
3. Check virtual environment: `which python` (should be in venv)
4. Review the configuration summary
5. Consult CLAUDE.md for project-specific guidance

---

**Happy Training!** 🎉
