#!/bin/bash
# setup.sh — Install dependencies and build pi_compute on Amazon Linux
# Usage: ./scripts/setup.sh [--cuda]
#
# This script:
# 1. Detects the Linux distribution (Amazon Linux 2, AL2023, Ubuntu)
# 2. Installs required dependencies (GMP, CMake, C++ compiler)
# 3. Optionally installs CUDA toolkit (if --cuda flag is passed)
# 4. Builds the project
# 5. Runs tests to verify everything works

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
ENABLE_CUDA=OFF

# Parse arguments
for arg in "$@"; do
    case $arg in
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--cuda]"
            echo "  --cuda    Enable CUDA GPU acceleration (requires NVIDIA GPU + drivers)"
            exit 0
            ;;
    esac
done

echo "============================================"
echo "  Pi Compute — Build Setup"
echo "============================================"
echo "  Project dir: $PROJECT_DIR"
echo "  CUDA:        $ENABLE_CUDA"
echo ""

# -------------------------------------------------------------------
# Step 1: Detect OS
# -------------------------------------------------------------------
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif [ -f /etc/system-release ]; then
        if grep -q "Amazon Linux" /etc/system-release; then
            echo "amzn"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

OS_ID=$(detect_os)
echo "[1/5] Detected OS: $OS_ID"

# -------------------------------------------------------------------
# Step 2: Install system dependencies
# -------------------------------------------------------------------
echo "[2/5] Installing dependencies..."

install_deps_amzn() {
    # Amazon Linux 2 or AL2023 — install only what we need (not the full "Development Tools" group)
    sudo yum install -y gcc gcc-c++ make cmake3 gmp-devel git

    # cmake3 may be installed as 'cmake3' on AL2
    if ! command -v cmake &>/dev/null; then
        if command -v cmake3 &>/dev/null; then
            sudo alternatives --install /usr/bin/cmake cmake /usr/bin/cmake3 1 2>/dev/null || \
                sudo ln -sf /usr/bin/cmake3 /usr/local/bin/cmake
            echo "  Linked cmake3 -> cmake"
        fi
    fi
}

install_deps_ubuntu() {
    sudo apt-get update
    # Install only the minimal set: compiler, make, cmake, GMP, git
    sudo apt-get install -y gcc g++ make cmake libgmp-dev git
}

install_deps_fedora() {
    # Install only what we need (not the full "Development Tools" group)
    sudo dnf install -y gcc gcc-c++ make cmake gmp-devel git
}

case "$OS_ID" in
    amzn|amazonlinux)
        install_deps_amzn
        ;;
    ubuntu|debian)
        install_deps_ubuntu
        ;;
    fedora|rhel|centos|rocky|almalinux)
        install_deps_fedora
        ;;
    *)
        echo "WARNING: Unknown OS '$OS_ID'. Attempting yum-based install..."
        install_deps_amzn
        ;;
esac

# -------------------------------------------------------------------
# Step 3: Install CUDA (optional)
# -------------------------------------------------------------------
if [ "$ENABLE_CUDA" = "ON" ]; then
    echo "[3/5] Checking CUDA installation..."

    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo "  CUDA already installed: $CUDA_VERSION"
    elif command -v nvidia-smi &>/dev/null; then
        echo "  NVIDIA driver found but CUDA toolkit not installed."
        echo "  Installing CUDA toolkit..."

        # For Amazon Linux with NVIDIA GPU
        if [ "$OS_ID" = "amzn" ] || [ "$OS_ID" = "amazonlinux" ]; then
            # Try to install from NVIDIA repo
            if [ ! -f /etc/yum.repos.d/cuda-rhel8.repo ]; then
                sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) 2>/dev/null || true
                sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo 2>/dev/null || \
                sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo 2>/dev/null || true
            fi
            sudo yum install -y cuda-toolkit 2>/dev/null || \
                echo "  WARNING: Could not auto-install CUDA. Please install manually."
        fi

        # Add CUDA to PATH if installed
        if [ -d /usr/local/cuda/bin ]; then
            export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
        fi
    else
        echo "  WARNING: No NVIDIA GPU/driver detected. Building without CUDA."
        ENABLE_CUDA=OFF
    fi
else
    echo "[3/5] Skipping CUDA (use --cuda to enable)"
fi

# -------------------------------------------------------------------
# Step 4: Validate dependencies
# -------------------------------------------------------------------
echo "[4/5] Validating dependencies..."

check_cmd() {
    if command -v "$1" &>/dev/null; then
        echo "  ✓ $1: $($1 --version 2>&1 | head -1)"
    else
        echo "  ✗ $1: NOT FOUND"
        return 1
    fi
}

DEPS_OK=true
check_cmd cmake || DEPS_OK=false
check_cmd g++ || check_cmd c++ || DEPS_OK=false

# Check GMP
if [ -f /usr/include/gmp.h ] || [ -f /usr/local/include/gmp.h ]; then
    echo "  ✓ gmp.h: found"
else
    echo "  ✗ gmp.h: NOT FOUND"
    DEPS_OK=false
fi

if [ "$ENABLE_CUDA" = "ON" ]; then
    check_cmd nvcc || { echo "  WARNING: nvcc not found, disabling CUDA"; ENABLE_CUDA=OFF; }
fi

if [ "$DEPS_OK" = false ]; then
    echo ""
    echo "ERROR: Missing dependencies. Please install them manually and re-run."
    exit 1
fi

# -------------------------------------------------------------------
# Step 5: Build
# -------------------------------------------------------------------
echo "[5/5] Building..."

cd "$PROJECT_DIR"
cmake -B "$BUILD_DIR" -DENABLE_CUDA=$ENABLE_CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "============================================"
echo "  Build complete!"
echo "============================================"
echo ""

# Run tests
echo "Running tests..."
cd "$BUILD_DIR"
ctest --output-on-failure

echo ""
echo "============================================"
echo "  All tests passed! ✓"
echo "============================================"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/src/pi_compute --digits 1000000 --verbose"
echo ""
echo "Quick benchmark:"
echo "  $BUILD_DIR/src/pi_compute --digits 10000000 --verbose --output /dev/null"
echo ""
