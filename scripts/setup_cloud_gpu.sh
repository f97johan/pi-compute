#!/bin/bash
# setup_cloud_gpu.sh — Quick setup for AWS GPU instances
# Usage: SSH into your instance, then:
#   git clone https://github.com/f97johan/pi-compute.git
#   cd pi-compute
#   ./scripts/setup_cloud_gpu.sh
#
# This script is specifically designed for AWS GPU instances
# (p3.2xlarge, p5.4xlarge, g7e.2xlarge, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "  Pi Compute — Cloud GPU Setup"
echo "============================================"
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""

    # Check CUDA
    if command -v nvcc &>/dev/null; then
        echo "CUDA toolkit: $(nvcc --version | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
        CUDA_FLAG="--cuda"
    else
        echo "CUDA toolkit not found. Installing..."
        CUDA_FLAG="--cuda"
    fi
else
    echo "No NVIDIA GPU detected. Building CPU-only."
    CUDA_FLAG=""
fi

echo ""
echo "Running setup..."
echo ""

# Run the main setup script
"$SCRIPT_DIR/setup.sh" $CUDA_FLAG

echo ""
echo "============================================"
echo "  Setup complete! Running benchmark..."
echo "============================================"
echo ""

# Run a quick benchmark
"$SCRIPT_DIR/benchmark.sh" 10000000

echo ""
echo "============================================"
echo "  Ready for pi computation!"
echo "============================================"
echo ""
echo "Examples:"
echo "  # Compute 10M digits"
echo "  ./build/src/pi_compute --digits 10000000 --verbose --output pi_10M.txt"
echo ""
echo "  # Compute 100M digits"
echo "  ./build/src/pi_compute --digits 100000000 --verbose --output pi_100M.txt"
echo ""
echo "  # Full benchmark"
echo "  ./scripts/benchmark.sh 100000000"
echo ""
