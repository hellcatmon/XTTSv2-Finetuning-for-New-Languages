# Installed Packages Information

This document describes all packages that are automatically installed by the setup scripts.

## 📦 Package Installation Steps

### 1. Core Dependencies (from requirements.txt)
The setup scripts install all packages from `requirements.txt` using UV package manager for Python 3.11.

**Installation method:**
```bash
uv pip install --python $(which python) -r requirements.txt
```

This includes:
- PyTorch and related packages
- TTS (Coqui TTS library)
- Audio processing libraries
- Training dependencies
- All other project dependencies

### 2. Additional Required Packages

The following packages are **automatically installed** after the main dependencies:

#### kagglehub
- **Purpose:** Access and download datasets from Kaggle
- **Use case:** Downloading training datasets from Kaggle repositories
- **Installation:** `uv pip install --python $(which python) kagglehub`
- **Verification:** `python -c "import kagglehub; print('kagglehub OK')"`

#### huggingface_hub
- **Purpose:** Access and download models/datasets from HuggingFace Hub
- **Use case:** Downloading pretrained models, sharing finetuned models
- **Installation:** `uv pip install --python $(which python) huggingface_hub`
- **Verification:** `python -c "import huggingface_hub; print('huggingface_hub OK')"`

#### ipykernel
- **Purpose:** Jupyter notebook kernel support
- **Use case:** Running code in Jupyter notebooks with Python 3.11
- **Installation:** `uv pip install --python $(which python) ipykernel`
- **Verification:** `jupyter kernelspec list` (should show python311_xtts)

## 🔍 Verification

After setup completes, the script automatically verifies all additional packages:

```bash
# Automatic verification during setup
✓ Found: kagglehub
✓ Found: huggingface_hub
✓ Found: ipykernel
```

### Manual Verification

You can manually verify package installation:

```bash
# Activate environment
source venv/bin/activate

# Check if packages are installed
pip list | grep kagglehub
pip list | grep huggingface-hub
pip list | grep ipykernel

# Test imports
python -c "import kagglehub; print('kagglehub:', kagglehub.__version__)"
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)"
python -c "import ipykernel; print('ipykernel:', ipykernel.__version__)"
```

## 💡 Usage Examples

### Using kagglehub

```python
import kagglehub

# Download a Kaggle dataset
dataset_path = kagglehub.dataset_download("username/dataset-name")
print(f"Dataset downloaded to: {dataset_path}")
```

### Using huggingface_hub

```python
from huggingface_hub import hf_hub_download, HfApi

# Download a model file
model_path = hf_hub_download(
    repo_id="anhnh2002/vnTTS",
    filename="model.pth"
)

# Upload your finetuned model
api = HfApi()
api.upload_folder(
    folder_path="checkpoints/GPT_XTTS_FT-latest",
    repo_id="username/my-xtts-model",
    repo_type="model"
)
```

### Using ipykernel in Jupyter

```python
# In Jupyter notebook:
# 1. Select kernel: "Python 3.11 (XTTSv2)"
# 2. Run your code

import sys
print(f"Python version: {sys.version}")
# Should show Python 3.11.x

# Run training
!python train_gpt_xtts.py --output_path=checkpoints/ ...
```

## 🎯 Why These Packages?

### kagglehub
- Enables easy access to Kaggle's extensive dataset library
- Simplifies dataset downloading for training
- Supports API-based authentication

### huggingface_hub
- Central hub for ML models and datasets
- Easy model sharing and collaboration
- Version control for models
- Community access to pretrained models

### ipykernel
- Essential for Jupyter notebook support
- Allows interactive development and testing
- Provides kernel isolation for Python 3.11
- Enables visualization and debugging

## 📝 Package Versions

The setup scripts install the latest compatible versions of these packages. To check installed versions:

```bash
pip show kagglehub
pip show huggingface-hub
pip show ipykernel
```

## 🔄 Updating Packages

To update the additional packages:

```bash
# Activate environment
source venv/bin/activate

# Update individual packages
pip install --upgrade kagglehub
pip install --upgrade huggingface_hub
pip install --upgrade ipykernel

# Or update all packages
pip install --upgrade kagglehub huggingface_hub ipykernel
```

## 🚨 Troubleshooting

### Package import fails

```bash
# Reinstall the package
pip install --force-reinstall kagglehub

# Verify Python version
python --version  # Should be 3.11.x

# Check package location
pip show kagglehub | grep Location
```

### Jupyter kernel not showing

```bash
# Re-register the kernel
python -m ipykernel install --user --name=python311_xtts --display-name="Python 3.11 (XTTSv2)"

# List all kernels
jupyter kernelspec list

# Remove old kernel if needed
jupyter kernelspec uninstall python311_xtts
```

### Version conflicts

```bash
# Check for conflicts
pip check

# Create fresh environment if needed
deactivate
rm -rf venv
./setup_env_py311.sh
```

## 📚 Additional Resources

- **kagglehub documentation:** https://github.com/Kaggle/kagglehub
- **huggingface_hub documentation:** https://huggingface.co/docs/huggingface_hub
- **ipykernel documentation:** https://ipykernel.readthedocs.io/

---

**Note:** All these packages are installed automatically by the setup scripts. You don't need to install them manually unless you're setting up the environment differently.
