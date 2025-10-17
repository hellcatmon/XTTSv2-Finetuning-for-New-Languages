#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo "Installing Packages to System Python 3.11"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    print_error "Python 3.11 not found!"
    print_info "Please install Python 3.11 first"
    exit 1
fi

PYTHON_VERSION=$(python3.11 --version 2>&1)
print_success "Found: $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

print_success "Found: requirements.txt"

# Step 1: Upgrade pip for Python 3.11 (optional)
echo ""
print_info "Step 1: Checking pip for Python 3.11..."

# Try to upgrade pip, but don't fail if it's managed by system package manager
python3.11 -m pip install --upgrade pip 2>/dev/null || print_warning "pip upgrade skipped (managed by system package manager)"

PIP_VERSION=$(python3.11 -m pip --version | awk '{print $2}')
print_success "pip version: $PIP_VERSION"

# Step 2: Install requirements.txt
echo ""
print_info "Step 2: Installing packages from requirements.txt..."
print_warning "This may take several minutes..."

python3.11 -m pip install -r requirements.txt

print_success "requirements.txt installed successfully"

# Step 3: Install additional packages
echo ""
print_info "Step 3: Installing additional packages..."

ADDITIONAL_PACKAGES=("kagglehub" "huggingface_hub" "ipykernel")

for package in "${ADDITIONAL_PACKAGES[@]}"; do
    print_info "Installing: $package"
    python3.11 -m pip install "$package"
done

print_success "Additional packages installed successfully"

# Step 4: Verify critical packages
echo ""
print_info "Step 4: Verifying critical packages..."

CRITICAL_PACKAGES=("torch" "transformers" "tokenizers" "pandas" "kagglehub" "huggingface_hub" "torchaudio" "soundfile")

VERIFICATION_PASSED=true

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python3.11 -c "import $package" 2>/dev/null; then
        print_success "✓ $package"
    else
        print_error "✗ $package"
        VERIFICATION_PASSED=false
    fi
done

echo ""
echo "=========================================="

if [ "$VERIFICATION_PASSED" = true ]; then
    print_success "🎉 All packages installed and verified!"
    echo ""
    print_info "You can now use python3.11 to run scripts:"
    echo "  python3.11 download_checkpoint.py --output_path /checkpoints/"
    echo "  python3.11 extend_vocab_config.py --output_path /checkpoints/ ..."
    echo ""
else
    print_error "Some packages failed to install or import"
    print_info "Please check the error messages above"
    exit 1
fi

echo "=========================================="
echo ""

# Show package count
PACKAGE_COUNT=$(python3.11 -m pip list --format=freeze | wc -l | tr -d ' ')
print_info "Total packages installed: $PACKAGE_COUNT"

echo ""
print_success "Installation complete!"
