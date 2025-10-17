# Downgrade Python 3.12 → 3.11 in Google Colab

## Solution: Install Python 3.11 using apt-get

Since Colab has Python 3.12 (incompatible), we need to install Python 3.11 manually.

---

## **Complete Solution (Copy All Cells)**

### **Cell 1: Install Python 3.11**
```python
# Install Python 3.11 from deadsnakes PPA
!apt-get update -qq
!apt-get install -y -qq software-properties-common
!add-apt-repository -y ppa:deadsnakes/ppa
!apt-get update -qq
!apt-get install -y -qq python3.11 python3.11-dev python3.11-distutils python3.11-venv

# Verify installation
!python3.11 --version
```

### **Cell 2: Install pip for Python 3.11**
```python
# Download and install pip for Python 3.11
!curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!python3.11 get-pip.py

# Verify pip
!python3.11 -m pip --version
```

### **Cell 3: Create Python 3.11 virtual environment**
```python
# Create virtual environment with Python 3.11
!python3.11 -m venv /content/venv311

# Activate virtual environment
import os
os.environ['PATH'] = '/content/venv311/bin:' + os.environ['PATH']
os.environ['VIRTUAL_ENV'] = '/content/venv311'

# Verify we're using Python 3.11 now
!python --version
!which python
```

### **Cell 4: Restart Python kernel**
```python
# IMPORTANT: After activating venv, restart the kernel
import os
os.kill(os.getpid(), 9)
```

**After kernel restarts, continue with Cell 5:**

### **Cell 5: Re-activate virtual environment**
```python
# Re-activate the virtual environment after kernel restart
import os
os.environ['PATH'] = '/content/venv311/bin:' + os.environ['PATH']
os.environ['VIRTUAL_ENV'] = '/content/venv311'

# Verify Python version
import sys
print(f"Python version: {sys.version}")
assert sys.version_info.major == 3 and sys.version_info.minor == 11, "Python 3.11 required!"
print("✅ Python 3.11 activated!")
```

### **Cell 6: Clone repository**
```python
!git clone -b feature/improvements https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git
%cd XTTSv2-Finetuning-for-New-Languages
```

### **Cell 7: Install PyTorch**
```python
# Upgrade pip first
!python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
!python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Cell 8: Install requirements**
```python
# Install all requirements
!python -m pip install -r requirements.txt
```

### **Cell 9: Verify installation**
```python
import sys
import torch

print("="*60)
print("Installation Verification")
print("="*60)
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
    print("Trying to install manually...")
    !python -m pip install trainer
    import trainer
    print("✅ Coqui trainer: Installed")

print("="*60)
print("✅ Setup complete! Ready to train.")
print("="*60)
```

### **Cell 10: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Create directories
!mkdir -p /content/drive/MyDrive/xtts_training/datasets
!mkdir -p /content/drive/MyDrive/xtts_training/checkpoints
```

---

## **Alternative: Simpler One-Cell Setup**

If the above seems complex, try this single-cell approach:

```python
# All-in-one setup
import os
import sys

# Install Python 3.11
!apt-get update -qq > /dev/null 2>&1
!apt-get install -y -qq software-properties-common > /dev/null 2>&1
!add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
!apt-get update -qq > /dev/null 2>&1
!apt-get install -y -qq python3.11 python3.11-dev python3.11-distutils python3.11-venv > /dev/null 2>&1

# Install pip for Python 3.11
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 > /dev/null 2>&1

# Set Python 3.11 as default for this session
os.environ['PYTHON'] = '/usr/bin/python3.11'

# Create symbolic link
!ln -sf /usr/bin/python3.11 /usr/local/bin/python
!ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# Verify
!python --version

print("✅ Python 3.11 installed and set as default!")
print("⚠️ Now restart runtime: Runtime → Restart runtime")
print("Then proceed with installing requirements")
```

**After running this cell:**
1. **Runtime → Restart runtime**
2. Then run:
```python
!python --version  # Should show 3.11
!git clone -b feature/improvements https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git
%cd XTTSv2-Finetuning-for-New-Languages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
```

---

## **Troubleshooting**

### Issue: Python still shows 3.12 after setup

**Solution:**
```python
# Force use Python 3.11
import sys
sys.executable = '/usr/bin/python3.11'

# Or use explicit path
!python3.11 -m pip install -r requirements.txt
```

### Issue: "No module named 'distutils'"

**Solution:**
```python
!apt-get install -y python3.11-distutils
```

### Issue: Virtual environment not activating

**Solution:** Use direct Python 3.11 path:
```python
# Instead of activating venv, use python3.11 directly
!python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!python3.11 -m pip install -r requirements.txt

# For training, use:
!python3.11 train_gpt_xtts_advanced.py --output_path checkpoints/ ...
```

### Issue: Kernel keeps restarting

**Solution:** This is normal when switching Python versions. Just proceed after restart.

---

## **Best Practice: Check Before Each Training**

Add this at the start of your training notebook:

```python
import sys
print(f"Python: {sys.version}")

# Ensure using Python 3.11
if sys.version_info.major != 3 or sys.version_info.minor != 11:
    print("⚠️ WARNING: Not using Python 3.11!")
    print("Using python3.11 explicitly for all commands...")
    PYTHON_CMD = "python3.11 -m"
else:
    print("✅ Python 3.11 confirmed")
    PYTHON_CMD = "python"
```

---

## **Summary**

1. Run Cell 1-4 to install and activate Python 3.11
2. After kernel restart, run Cell 5-10
3. You're ready to train!

**Key commands to remember:**
- `!python3.11 --version` - Check Python 3.11 is installed
- `!python3.11 -m pip install ...` - Install packages with Python 3.11
- `!python3.11 script.py` - Run scripts with Python 3.11

This will work without pyenv or conda! 🚀
