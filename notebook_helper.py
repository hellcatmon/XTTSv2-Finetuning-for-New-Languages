"""
Helper script to set up the virtual environment in Jupyter notebooks.

Usage in notebook:
    %run /XTTSv2-Finetuning-for-New-Languages/notebook_helper.py

This will automatically configure the notebook to use the venv packages.
"""

import sys
import os

# Path to the virtual environment
VENV_PATH = "/XTTSv2-Finetuning-for-New-Languages/venv"
SITE_PACKAGES = f"{VENV_PATH}/lib/python3.11/site-packages"

def setup_venv():
    """Add virtual environment to Python path"""
    if os.path.exists(SITE_PACKAGES):
        # Insert at the beginning to override system packages
        if SITE_PACKAGES not in sys.path:
            sys.path.insert(0, SITE_PACKAGES)
        print(f"✓ Virtual environment configured: {VENV_PATH}")
        print(f"✓ Python: {sys.executable}")
        print(f"✓ Version: {sys.version.split()[0]}")
        return True
    else:
        print(f"✗ Virtual environment not found at: {VENV_PATH}")
        print(f"  Please run standalone_setup.sh first")
        return False

def verify_packages():
    """Verify critical packages are available"""
    critical_packages = [
        "torch",
        "transformers",
        "tokenizers",
        "pandas",
        "torchaudio",
        "soundfile"
    ]

    print("\nVerifying packages:")
    all_ok = True
    for package in critical_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            all_ok = False

    if all_ok:
        print("\n✓ All critical packages available!")
    else:
        print("\n✗ Some packages are missing. Please run standalone_setup.sh")

    return all_ok

# Auto-run when loaded with %run
if __name__ == "__main__":
    print("=" * 50)
    print("XTTSv2 Notebook Helper")
    print("=" * 50)
    if setup_venv():
        verify_packages()
    print("=" * 50)
