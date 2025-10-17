# Python 3.11 Setup in Google Colab

## ⚠️ Important: Colab's Default Python 3.10 Works!

**Good News:** Google Colab uses Python 3.10 by default, which is **fully compatible** with this project. You **DO NOT need to downgrade** to Python 3.11!

### Check Your Python Version First

```python
import sys
print(f"Python version: {sys.version}")

# Should show: Python 3.10.x
# This is compatible! ✅
```

If you see Python 3.10.x, **you can skip the rest of this guide** and proceed directly with:

```python
!pip install -r requirements.txt
```

---

## Only If You Absolutely Need Python 3.11

If for some reason you need exactly Python 3.11 (you probably don't), here are the options:

### Option 1: Use Conda (Recommended for Colab)

```python
# Install conda packages for Python 3.11
!conda install -y python=3.11 -c conda-forge

# Verify
import sys
print(f"Python version: {sys.version}")
```

### Option 2: Manual Python 3.11 Installation

```python
# Install Python 3.11 from deadsnakes PPA
!apt-get update
!apt-get install -y software-properties-common
!add-apt-repository -y ppa:deadsnakes/ppa
!apt-get update
!apt-get install -y python3.11 python3.11-dev python3.11-distutils

# Install pip for Python 3.11
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Verify installation
!python3.11 --version

# Create alias (for current session only)
import sys
sys.executable = '/usr/bin/python3.11'
```

### Option 3: Use Python 3.11 with Virtual Environment

```python
# Install Python 3.11
!apt-get update
!apt-get install -y python3.11 python3.11-venv python3.11-dev

# Create virtual environment
!python3.11 -m venv /content/venv311

# Activate (requires restart)
import os
os.environ['PATH'] = '/content/venv311/bin:' + os.environ['PATH']

# Verify
!which python
!python --version
```

---

## ⚠️ Problems with Changing Python in Colab

Changing Python version in Colab is **not recommended** because:

1. Colab's system packages are built for Python 3.10
2. Many system libraries may break
3. Jupyter kernel needs restart
4. Can cause unpredictable errors

---

## ✅ Recommended Solution: Just Use Python 3.10!

**The project works perfectly with Python 3.10.** Here's the correct setup:

```python
# 1. Check Python version (should be 3.10.x)
import sys
print(f"Python version: {sys.version}")

# 2. Clone repository
!git clone -b feature/improvements https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git
%cd XTTSv2-Finetuning-for-New-Languages

# 3. Install PyTorch with CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install requirements
!pip install -r requirements.txt

# 5. Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import trainer
    print("✅ All packages installed successfully!")
except ImportError as e:
    print(f"❌ Error: {e}")
```

---

## If You Get "trainer" Package Errors

The `trainer` package from Coqui TTS works with Python 3.10. If you get errors:

```python
# Try installing trainer explicitly
!pip install TTS

# Or install from requirements
!pip install -r requirements.txt --no-cache-dir
```

---

## Complete Working Setup for Colab (Copy & Paste)

```python
# ========================================
# STEP 1: Verify Environment
# ========================================
import sys
print(f"Python version: {sys.version}")
assert sys.version_info.major == 3 and sys.version_info.minor == 10, "Python 3.10 required"
print("✅ Python 3.10 detected - perfect for this project!")

# ========================================
# STEP 2: Check GPU
# ========================================
!nvidia-smi

# ========================================
# STEP 3: Clone Repository
# ========================================
!git clone -b feature/improvements https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git
%cd XTTSv2-Finetuning-for-New-Languages

# ========================================
# STEP 4: Install PyTorch (IMPORTANT: Do this first!)
# ========================================
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ========================================
# STEP 5: Install Requirements
# ========================================
!pip install -r requirements.txt

# ========================================
# STEP 6: Install Additional Packages
# ========================================
!pip install tensorboard deepspeed

# ========================================
# STEP 7: Verify Installation
# ========================================
import torch
print(f"\n{'='*60}")
print("Installation Verification")
print(f"{'='*60}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    import trainer
    print("✅ Coqui trainer: Installed")
except ImportError:
    print("❌ Coqui trainer: NOT installed")

try:
    import torchaudio
    print("✅ torchaudio: Installed")
except ImportError:
    print("❌ torchaudio: NOT installed")

print(f"{'='*60}")
print("✅ Setup complete! Ready to train.")
print(f"{'='*60}\n")

# ========================================
# STEP 8: Mount Google Drive (Optional)
# ========================================
from google.colab import drive
drive.mount('/content/drive')
```

---

## Why Python 3.10 Works

The project requires:
- Python >= 3.6
- Python < 3.12

Python 3.10 is in this range and works perfectly! The error you saw earlier was likely about Python 3.12, not 3.10.

### Compatibility Matrix

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.8 | ⚠️ Works | Old, not recommended |
| 3.9 | ✅ Works | Good |
| **3.10** | **✅ Perfect** | **Colab default, recommended** |
| 3.11 | ✅ Works | Compatible, but unnecessary |
| 3.12 | ❌ Fails | trainer package incompatible |
| 3.13 | ❌ Fails | Not supported |

---

## Troubleshooting

### Error: "Could not find a version that satisfies the requirement trainer"

**This means you have Python 3.12+, not 3.10!**

Check your Python version:
```python
import sys
print(sys.version_info)
```

If it shows Python 3.12, restart your Colab runtime:
1. Runtime → Restart runtime
2. Check version again (should be 3.10)
3. Run installation again

### Error: "No module named 'torch'"

Install PyTorch first:
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "CUDA not available"

Ensure GPU runtime is enabled:
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save

---

## Summary

**You DO NOT need Python 3.11!**

Colab's Python 3.10 is perfect. Just use the "Complete Working Setup" above.

If you still get errors, share the exact error message and I'll help you debug it.
