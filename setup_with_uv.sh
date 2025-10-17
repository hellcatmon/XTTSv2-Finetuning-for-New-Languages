#!/bin/bash
set -e

echo "🚀 Setting up XTTSv2 Finetuning environment with uv (fast package manager)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ uv version: $(uv --version)"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install packages using uv (much faster than pip)
echo "📥 Installing dependencies with uv..."
uv pip install -r requirements.txt

# Install spacy language model for Japanese
echo "📥 Installing spacy Japanese language model..."
python -m spacy download ja_core_news_sm

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Speed comparison: uv is typically 10-100x faster than pip! 🚀"
