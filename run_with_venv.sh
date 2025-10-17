#!/bin/bash
# Simple wrapper to run Python scripts with the virtual environment

SCRIPT_DIR="/XTTSv2-Finetuning-for-New-Languages"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"

# Check if venv Python exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at $VENV_PYTHON"
    echo "Please run standalone_setup.sh first"
    exit 1
fi

# Run the command with venv Python
"$VENV_PYTHON" "$@"
