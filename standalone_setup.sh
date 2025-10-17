#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "XTTSv2 Standalone Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Variables for summary
START_TIME=$(date +%s)
PROJECT_DIR="XTTSv2-Finetuning-for-New-Languages"
GIT_REPO="https://github.com/hellcatmon/XTTSv2-Finetuning-for-New-Languages.git"
GIT_BRANCH="feature/improvements"

# Check if running on Linux (for apt-get commands)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    IS_LINUX=true
    print_info "Detected Linux system"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    IS_LINUX=false
    print_info "Detected macOS system"
else
    IS_LINUX=false
    print_warning "Unknown OS: $OSTYPE. Will attempt to proceed..."
fi

# Step 1: Check if Python 3.11 is already installed
echo ""
echo "Step 1: Checking for Python 3.11..."
echo "--------------------------------------"

if command -v python3.11 &> /dev/null; then
    PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
    print_success "Python 3.11 is already installed: $PYTHON_VERSION"
    PYTHON_CMD="python3.11"
else
    print_warning "Python 3.11 not found. Installing..."

    if [ "$IS_LINUX" = true ]; then
        echo "Installing Python 3.11 on Linux..."

        # Update package list
        print_info "Updating package list..."
        apt-get update -qq

        # Install software-properties-common if not present
        print_info "Installing software-properties-common..."
        apt-get install -y -qq software-properties-common

        # Add deadsnakes PPA for Python 3.11
        print_info "Adding deadsnakes PPA..."
        add-apt-repository -y ppa:deadsnakes/ppa

        # Update package list again
        apt-get update -qq

        # Install Python 3.11 and related packages
        print_info "Installing Python 3.11 and dependencies..."
        apt-get install -y -qq python3.11 python3.11-dev python3.11-distutils python3.11-venv

        print_success "Python 3.11 installed successfully"
        PYTHON_CMD="python3.11"
    else
        print_error "Automatic Python 3.11 installation is only supported on Linux"
        print_info "Please install Python 3.11 manually:"
        print_info "  - macOS: brew install python@3.11"
        print_info "  - Windows: Download from python.org"
        exit 1
    fi
fi

# Verify Python 3.11 installation
if ! command -v python3.11 &> /dev/null; then
    print_error "Python 3.11 installation failed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
print_success "Using Python version: $PYTHON_VERSION"

# Step 2: Install uv package manager
echo ""
echo "Step 2: Installing uv package manager..."
echo "--------------------------------------"

if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1 | awk '{print $2}')
    print_success "uv is already installed: $UV_VERSION"
else
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version 2>&1 | awk '{print $2}')
        print_success "uv installed successfully: $UV_VERSION"
    else
        print_error "uv installation failed"
        print_info "Falling back to pip installation method..."
        UV_AVAILABLE=false
    fi
fi

# Step 3: Clone the repository
echo ""
echo "Step 3: Cloning XTTSv2 repository..."
echo "--------------------------------------"

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Directory '$PROJECT_DIR' already exists"
    read -p "Do you want to remove it and clone fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing directory..."
        rm -rf "$PROJECT_DIR"
        CLONE_FRESH=true
    else
        print_info "Using existing directory"
        CLONE_FRESH=false
    fi
else
    CLONE_FRESH=true
fi

if [ "$CLONE_FRESH" = true ]; then
    print_info "Cloning repository (branch: $GIT_BRANCH)..."
    git clone --branch "$GIT_BRANCH" "$GIT_REPO"
    print_success "Repository cloned successfully"
else
    print_info "Checking if directory is a git repository..."
    if [ -d "$PROJECT_DIR/.git" ]; then
        cd "$PROJECT_DIR"
        CURRENT_BRANCH=$(git branch --show-current)
        print_info "Current branch: $CURRENT_BRANCH"

        if [ "$CURRENT_BRANCH" != "$GIT_BRANCH" ]; then
            print_warning "Not on branch '$GIT_BRANCH'"
            read -p "Do you want to checkout '$GIT_BRANCH'? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git fetch origin
                git checkout "$GIT_BRANCH"
                git pull origin "$GIT_BRANCH"
                print_success "Switched to branch '$GIT_BRANCH'"
            fi
        else
            print_info "Pulling latest changes..."
            git pull origin "$GIT_BRANCH"
            print_success "Repository updated"
        fi
        cd ..
    else
        print_error "$PROJECT_DIR exists but is not a git repository"
        exit 1
    fi
fi

# Change to project directory
cd "$PROJECT_DIR"
INSTALL_DIR=$(pwd)
print_success "Working directory: $INSTALL_DIR"

# Step 4: Create virtual environment with Python 3.11
echo ""
echo "Step 4: Creating virtual environment..."
echo "--------------------------------------"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment with Python 3.11..."
    python3.11 -m venv "$VENV_DIR"
    print_success "Virtual environment created at $VENV_DIR"
fi

# Step 5: Activate virtual environment
echo ""
echo "Step 5: Activating virtual environment..."
echo "--------------------------------------"

source "$VENV_DIR/bin/activate"

# Verify we're using Python 3.11 in the venv
VENV_PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ $VENV_PYTHON_VERSION == 3.11.* ]]; then
    print_success "Virtual environment activated with Python $VENV_PYTHON_VERSION"
else
    print_error "Virtual environment is not using Python 3.11 (found: $VENV_PYTHON_VERSION)"
    exit 1
fi

# Step 6: Upgrade pip
echo ""
echo "Step 6: Upgrading pip..."
echo "--------------------------------------"
python -m pip install --upgrade pip -q
PIP_VERSION=$(pip --version | awk '{print $2}')
print_success "pip upgraded successfully: $PIP_VERSION"

# Step 7: Install dependencies with uv
echo ""
echo "Step 7: Installing project dependencies with uv..."
echo "--------------------------------------"

if [ -f "requirements.txt" ]; then
    if command -v uv &> /dev/null && [ "$UV_AVAILABLE" != false ]; then
        print_info "Installing dependencies with uv (10-100x faster than pip)..."
        print_info "This may take a few minutes on first run..."

        # Use uv with explicit Python 3.11 interpreter
        print_info "Using Python 3.11 interpreter: $(which python)"
        uv pip install --python $(which python) -r requirements.txt

        print_success "Dependencies installed successfully with uv"
        INSTALL_METHOD="uv"
    else
        print_warning "uv not available, falling back to pip..."
        print_info "Installing dependencies with pip (this may take longer)..."
        pip install -r requirements.txt
        print_success "Dependencies installed successfully with pip"
        INSTALL_METHOD="pip"
    fi

    # Count installed packages
    PACKAGE_COUNT=$(pip list --format=freeze | wc -l | tr -d ' ')

    # Verify packages are installed for Python 3.11
    print_info "Verifying packages are installed for Python 3.11..."
    ACTUAL_PYTHON=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $ACTUAL_PYTHON == "3.11" ]]; then
        print_success "Confirmed: Packages installed for Python $ACTUAL_PYTHON"
    else
        print_error "Warning: Packages may be installed for Python $ACTUAL_PYTHON instead of 3.11"
    fi
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Step 8: Register Python 3.11 for Jupyter notebooks
echo ""
echo "Step 8: Registering Python 3.11 kernel for Jupyter..."
echo "--------------------------------------"

# Check if ipykernel is installed, if not install it
if ! python -c "import ipykernel" &> /dev/null; then
    print_info "Installing ipykernel for Jupyter support..."
    if command -v uv &> /dev/null && [ "$UV_AVAILABLE" != false ]; then
        uv pip install --python $(which python) ipykernel
    else
        pip install ipykernel
    fi
fi

# Register the kernel
print_info "Registering Python 3.11 kernel as 'python311_xtts'..."
python -m ipykernel install --user --name=python311_xtts --display-name="Python 3.11 (XTTSv2)"
print_success "Jupyter kernel registered successfully"
print_info "You can now select 'Python 3.11 (XTTSv2)' in Jupyter notebooks"

# Verify python3.11 command is available
echo ""
print_info "Verifying python3.11 command availability..."
if command -v python3.11 &> /dev/null; then
    PYTHON311_PATH=$(which python3.11)
    print_success "python3.11 is available at: $PYTHON311_PATH"
    print_success "python3.11 version: $(python3.11 --version)"
else
    print_error "python3.11 command not found in PATH!"
    exit 1
fi

# Calculate setup time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))
ELAPSED_SEC=$((ELAPSED_TIME % 60))

# Step 9: Verify installation
echo ""
echo "Step 9: Verifying installation..."
echo "--------------------------------------"

# Check for key project files
VERIFICATION_PASSED=true

if [ -f "train_gpt_xtts.py" ]; then
    print_success "Found: train_gpt_xtts.py"
else
    print_error "Missing: train_gpt_xtts.py"
    VERIFICATION_PASSED=false
fi

if [ -f "download_checkpoint.py" ]; then
    print_success "Found: download_checkpoint.py"
else
    print_error "Missing: download_checkpoint.py"
    VERIFICATION_PASSED=false
fi

if [ -f "extend_vocab_config_improved.py" ]; then
    print_success "Found: extend_vocab_config_improved.py"
else
    print_warning "Missing: extend_vocab_config_improved.py (may be in older branch)"
fi

if [ -d "TTS" ]; then
    print_success "Found: TTS directory"
else
    print_error "Missing: TTS directory"
    VERIFICATION_PASSED=false
fi

# Check CUDA availability
print_info "Checking CUDA availability..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    CUDA_AVAILABLE="Yes"
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "N/A")
    print_success "CUDA is available"
else
    CUDA_AVAILABLE="No"
    CUDA_VERSION="N/A"
    GPU_COUNT="0"
    GPU_NAME="N/A"
    print_warning "CUDA not available (CPU-only mode)"
fi

# Get Git information
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A")
GIT_CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "N/A")

# Configuration Summary
echo ""
echo "=========================================="
print_header "        SETUP COMPLETE!        "
echo "=========================================="
echo ""

print_header "📊 CONFIGURATION SUMMARY"
echo ""

print_header "🐍 Python Environment:"
echo "  Python Version:        $(python --version | awk '{print $2}')"
echo "  Python Path:           $(which python)"
echo "  python3.11 Command:    $(which python3.11)"
echo "  Virtual Environment:   $VENV_DIR"
echo "  Pip Version:           $PIP_VERSION"
echo "  Jupyter Kernel:        python311_xtts (Python 3.11 (XTTSv2))"

echo ""
print_header "📦 Package Management:"
if command -v uv &> /dev/null; then
    echo "  uv Version:            $(uv --version | awk '{print $2}')"
    echo "  Installation Method:   $INSTALL_METHOD"
else
    echo "  uv Version:            Not installed"
    echo "  Installation Method:   pip"
fi
echo "  Packages Installed:    $PACKAGE_COUNT"

echo ""
print_header "📁 Project Information:"
echo "  Repository:            $GIT_REPO"
echo "  Branch:                $GIT_CURRENT_BRANCH"
echo "  Commit:                $GIT_COMMIT"
echo "  Install Directory:     $INSTALL_DIR"

echo ""
print_header "🖥️  Hardware & CUDA:"
echo "  Operating System:      $OSTYPE"
echo "  CUDA Available:        $CUDA_AVAILABLE"
if [ "$CUDA_AVAILABLE" = "Yes" ]; then
    echo "  CUDA Version:          $CUDA_VERSION"
    echo "  GPU Count:             $GPU_COUNT"
    echo "  GPU Model:             $GPU_NAME"
fi

echo ""
print_header "⏱️  Installation Time:"
echo "  Total Time:            ${ELAPSED_MIN}m ${ELAPSED_SEC}s"

echo ""
echo "=========================================="
print_header "        NEXT STEPS        "
echo "=========================================="
echo ""

if [ "$VERIFICATION_PASSED" = true ]; then
    print_success "All critical files verified!"
else
    print_error "Some files are missing. Please check the repository."
fi

echo ""
print_info "To activate the environment:"
echo "  cd $INSTALL_DIR"
echo "  source $VENV_DIR/bin/activate"
echo ""

print_info "To deactivate the environment:"
echo "  deactivate"
echo ""

print_info "To use in Jupyter Notebooks:"
echo "  1. Start Jupyter: jupyter notebook or jupyter lab"
echo "  2. Select kernel: 'Python 3.11 (XTTSv2)'"
echo "  3. Or use python3.11 command directly in notebook cells"
echo ""

print_header "🚀 Quick Start Commands:"
echo ""
echo "1. Download pretrained model:"
echo "   python download_checkpoint.py --output_path checkpoints/"
echo ""
echo "2. Extend vocabulary for new language:"
echo "   python extend_vocab_config_improved.py \\"
echo "     --output_path=checkpoints/ \\"
echo "     --metadata_path=datasets/metadata_train.csv \\"
echo "     --language=vi"
echo ""
echo "3. Train GPT model:"
echo "   CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \\"
echo "     --output_path=checkpoints/ \\"
echo "     --metadatas=datasets/metadata_train.csv,datasets/metadata_eval.csv,vi \\"
echo "     --num_epochs=5 --batch_size=8"
echo ""

print_header "📚 Documentation:"
echo "  CLAUDE.md              - Project overview and instructions"
echo "  P1_IMPROVEMENTS_README.md - Recent improvements (P1)"
echo "  README.md              - Original README"
echo ""

print_success "Setup completed successfully! Happy training! 🎉"
echo ""
