# Using Virtual Environment in Jupyter Notebooks

This guide explains how to use the virtual environment created by `standalone_setup.sh` in your Jupyter notebooks.

## Option 1: Use the Registered Jupyter Kernel (Recommended)

The `standalone_setup.sh` script already registered a Jupyter kernel called **"Python 3.11 (XTTSv2)"**.

### Steps:
1. Start your Jupyter notebook server
2. Open your notebook
3. Click on **Kernel → Change Kernel**
4. Select **"Python 3.11 (XTTSv2)"**

This kernel automatically uses the virtual environment with all installed packages.

## Option 2: Activate venv in Notebook Cells

If you want to use the venv directly in notebook cells, add this at the top of your notebook:

```python
import sys
sys.path.insert(0, '/XTTSv2-Finetuning-for-New-Languages/venv/lib/python3.11/site-packages')
```

Then run your scripts normally:

```python
!cd /XTTSv2-Finetuning-for-New-Languages && python download_checkpoint.py --output_path /checkpoints/
```

## Option 3: Use Shell Commands with venv Activation

Prefix your commands with venv activation:

```python
!source /XTTSv2-Finetuning-for-New-Languages/venv/bin/activate && \
 cd /XTTSv2-Finetuning-for-New-Languages && \
 python download_checkpoint.py --output_path /checkpoints/
```

## Option 4: Use Python Directly from venv

```python
!/XTTSv2-Finetuning-for-New-Languages/venv/bin/python \
  /XTTSv2-Finetuning-for-New-Languages/download_checkpoint.py \
  --output_path /checkpoints/
```

## Verify Installation

Check if packages are available:

```python
# In a notebook cell
import sys
print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")

# Test imports
import torch
import transformers
import tokenizers
print("✓ All packages imported successfully!")
```

## Recommended Approach

**Use Option 1 (Jupyter Kernel)** - it's the cleanest and most reliable approach. The kernel was specifically registered for this purpose during setup.
