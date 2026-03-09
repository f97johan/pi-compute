#!/bin/bash
# benchmark.sh — Run benchmarks at various digit counts
# Usage: ./scripts/benchmark.sh [max_digits]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PI_COMPUTE="$PROJECT_DIR/build/src/pi_compute"

if [ ! -f "$PI_COMPUTE" ]; then
    echo "ERROR: pi_compute not found. Run ./scripts/setup.sh first."
    exit 1
fi

MAX_DIGITS=${1:-10000000}

echo "============================================"
echo "  Pi Compute — Benchmark Suite"
echo "============================================"
echo "  Max digits: $MAX_DIGITS"
echo "  Binary:     $PI_COMPUTE"
echo ""

# System info
echo "System Info:"
echo "  CPU: $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
echo "  Cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
echo "  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB\n", $1/1073741824}' || echo 'unknown')"

if command -v nvidia-smi &>/dev/null; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
fi
echo ""

echo "Benchmark Results:"
echo "-------------------------------------------"
printf "%-15s %-12s %-10s\n" "Digits" "Time (s)" "Terms"
echo "-------------------------------------------"

for digits in 1000 10000 100000 1000000; do
    if [ "$digits" -gt "$MAX_DIGITS" ]; then
        break
    fi
    output=$($PI_COMPUTE --digits $digits --output /dev/null --verbose 2>&1)
    time_s=$(echo "$output" | grep "Done in" | sed 's/.*Done in //' | sed 's/ seconds.*//')
    terms=$(echo "$output" | grep "Terms needed" | sed 's/.*Terms needed: //')
    printf "%-15s %-12s %-10s\n" "$(printf "%'d" $digits)" "$time_s" "$terms"
done

# Larger benchmarks
for digits in 5000000 10000000 50000000 100000000; do
    if [ "$digits" -gt "$MAX_DIGITS" ]; then
        break
    fi
    echo ""
    echo "Computing $(printf "%'d" $digits) digits..."
    output=$($PI_COMPUTE --digits $digits --output /dev/null --verbose 2>&1)
    time_s=$(echo "$output" | grep "Done in" | sed 's/.*Done in //' | sed 's/ seconds.*//')
    terms=$(echo "$output" | grep "Terms needed" | sed 's/.*Terms needed: //')
    printf "%-15s %-12s %-10s\n" "$(printf "%'d" $digits)" "$time_s" "$terms"
done

echo "-------------------------------------------"
echo ""
echo "Done!"
