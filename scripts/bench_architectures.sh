#!/usr/bin/env bash
#
# bench_architectures.sh — Automated cross-architecture pi benchmark
#
# Launches one instance per architecture (Intel, AMD, Graviton3, Graviton4),
# builds the project, runs benchmarks, collects results, and terminates.
#
# Usage:
#   ./scripts/bench_architectures.sh [--digits N] [--dry-run] [--key-name KEY] [--region REGION]
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - An EC2 key pair (for SSH access)
#   - Default VPC with internet access
#
# Cost estimate: ~$2-4 total (4 instances × ~30 min × ~$0.17/hr each)

set -eo pipefail

# ============================================================
# Configuration
# ============================================================

DIGITS=100000000          # 100M digits by default (~1-3 min per arch)
DRY_RUN=false
KEY_NAME=""
KEY_FILE=""
REGION="us-east-1"
SUBNET_ID=""
SECURITY_GROUP=""
GITHUB_REPO="https://github.com/f97johan/pi-compute.git"
RESULTS_DIR="benchmark-results"

# Instance configs: "label:instance_type:ami_arch"
CONFIGS=(
    "intel:c7i.2xlarge:x86_64"
    "amd:c7a.2xlarge:x86_64"
    "graviton3:c7g.2xlarge:arm64"
    "graviton4:c8g.2xlarge:arm64"
)

# ============================================================
# Parse arguments
# ============================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --digits)      DIGITS="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --key-name)    KEY_NAME="$2"; shift 2 ;;
        --key-file)    KEY_FILE="$2"; shift 2 ;;
        --region)      REGION="$2"; shift 2 ;;
        --subnet-id)   SUBNET_ID="$2"; shift 2 ;;
        --sg)          SECURITY_GROUP="$2"; shift 2 ;;
        --repo)        GITHUB_REPO="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--digits N] [--dry-run] [--key-name KEY] [--key-file PATH] [--region REGION]"
            echo ""
            echo "Options:"
            echo "  --digits N       Number of pi digits to compute (default: 100000000)"
            echo "  --dry-run        Print commands without executing"
            echo "  --key-name KEY   EC2 key pair name (as it appears in AWS)"
            echo "  --key-file PATH  Path to the .pem private key file for SSH"
            echo "  --region REGION  AWS region (default: us-east-1)"
            echo "  --subnet-id ID   Subnet ID (optional, uses default VPC)"
            echo "  --sg SG_ID       Security group ID (optional)"
            echo "  --repo URL       Git repo URL"
            echo ""
            echo "Both --key-name and --key-file are required unless --dry-run."
            echo "  --key-name is the name in AWS (used for ec2 run-instances)"
            echo "  --key-file is the local .pem file path (used for SSH/SCP)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$DRY_RUN" = "false" ]; then
    if [ -z "$KEY_NAME" ] || [ -z "$KEY_FILE" ]; then
        echo "ERROR: Both --key-name and --key-file are required (unless --dry-run)"
        echo ""
        echo "  --key-name  The EC2 key pair name (as shown in AWS Console)"
        echo "  --key-file  Path to the corresponding .pem private key file"
        echo ""
        echo "Example:"
        echo "  # Create a new key pair:"
        echo "  aws ec2 create-key-pair --key-name pi-bench --query 'KeyMaterial' --output text > ~/.ssh/pi-bench.pem"
        echo "  chmod 400 ~/.ssh/pi-bench.pem"
        echo ""
        echo "  # Run the benchmark:"
        echo "  $0 --key-name pi-bench --key-file ~/.ssh/pi-bench.pem"
        exit 1
    fi
    if [ ! -f "$KEY_FILE" ]; then
        echo "ERROR: Key file not found: $KEY_FILE"
        exit 1
    fi
fi

mkdir -p "$RESULTS_DIR"

# ============================================================
# Helper functions
# ============================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Arrays to track instances (indexed, not associative — bash 3.2 compatible)
LABELS=()
INSTANCE_IDS=()
INSTANCE_IPS=()

lookup_ami() {
    local arch="$1"
    aws ec2 describe-images \
        --region "$REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-*-server-*" \
            "Name=architecture,Values=$arch" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text
}

# ============================================================
# User data script (cloud-init, runs on boot)
# ============================================================

generate_userdata() {
    cat <<EOF
#!/bin/bash
set -ex
exec > /var/log/pi-benchmark.log 2>&1

# Install dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake libgmp-dev git time

# Clone and build
cd /home/ubuntu
git clone ${GITHUB_REPO} pi-compute
cd pi-compute
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j\$(nproc)

# Run tests
./build/tests/pi_tests

# System info
echo "=== BENCHMARK START ==="
echo "Architecture: \$(uname -m)"
echo "CPU: \$(lscpu | grep 'Model name' | head -1 | sed 's/.*: *//')"
echo "Cores: \$(nproc)"
echo "RAM: \$(free -h | awk '/Mem:/ {print \$2}')"
echo "GMP version: \$(dpkg -s libgmp-dev | grep Version)"
echo "Digits: ${DIGITS}"
echo ""

# Benchmark: 3 runs
for run in 1 2 3; do
    echo "--- Run \$run ---"
    ./build/src/pi_compute \\
        --digits ${DIGITS} \\
        --verbose \\
        --output /dev/null
    echo ""
done

echo "=== BENCHMARK COMPLETE ==="
touch /home/ubuntu/benchmark_done
EOF
}

# ============================================================
# Cleanup on exit
# ============================================================

cleanup() {
    log "Cleaning up..."
    for i in $(seq 0 $((${#INSTANCE_IDS[@]} - 1))); do
        local iid="${INSTANCE_IDS[$i]}"
        local label="${LABELS[$i]}"
        if [ -n "$iid" ] && [ "$iid" != "i-dryrun" ]; then
            log "  Terminating $label: $iid"
            aws ec2 terminate-instances --region "$REGION" --instance-ids "$iid" --output text 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

# ============================================================
# Main
# ============================================================

log "Pi Architecture Benchmark"
log "Digits: $DIGITS"
log "Region: $REGION"
log "Architectures: ${CONFIGS[*]}"
log ""

# Step 1: Launch instances
log "Launching instances..."

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r label itype ami_arch <<< "$config"
    LABELS+=("$label")

    log "  $label: $itype ($ami_arch)"

    if [ "$DRY_RUN" = "true" ]; then
        INSTANCE_IDS+=("i-dryrun")
        INSTANCE_IPS+=("0.0.0.0")
        log "    [DRY-RUN] Would launch $itype"
        continue
    fi

    # Look up AMI
    ami=$(lookup_ami "$ami_arch")
    log "    AMI: $ami"

    # Generate user data
    userdata_b64=$(generate_userdata | base64)

    # Build launch args
    launch_args="--region $REGION"
    launch_args="$launch_args --image-id $ami"
    launch_args="$launch_args --instance-type $itype"
    launch_args="$launch_args --key-name $KEY_NAME"
    launch_args="$launch_args --user-data $userdata_b64"
    launch_args="$launch_args --tag-specifications ResourceType=instance,Tags=[{Key=Name,Value=pi-bench-$label},{Key=Project,Value=pi-compute}]"
    launch_args="$launch_args --instance-initiated-shutdown-behavior terminate"
    launch_args="$launch_args --query Instances[0].InstanceId"
    launch_args="$launch_args --output text"

    if [ -n "$SUBNET_ID" ]; then
        launch_args="$launch_args --subnet-id $SUBNET_ID"
    fi
    if [ -n "$SECURITY_GROUP" ]; then
        launch_args="$launch_args --security-group-ids $SECURITY_GROUP"
    fi

    iid=$(eval aws ec2 run-instances $launch_args)
    INSTANCE_IDS+=("$iid")
    INSTANCE_IPS+=("")  # placeholder, filled after running
    log "    Instance: $iid"
done

# Step 2: Wait for instances and get IPs
if [ "$DRY_RUN" = "false" ]; then
    log ""
    log "Waiting for instances to start..."
    for i in $(seq 0 $((${#INSTANCE_IDS[@]} - 1))); do
        iid="${INSTANCE_IDS[$i]}"
        label="${LABELS[$i]}"
        aws ec2 wait instance-running --region "$REGION" --instance-ids "$iid"
        ip=$(aws ec2 describe-instances \
            --region "$REGION" \
            --instance-ids "$iid" \
            --query 'Reservations[0].Instances[0].PublicIpAddress' \
            --output text)
        INSTANCE_IPS[$i]="$ip"
        log "  $label: $ip"
    done

    # Step 3: Wait for benchmarks
    log ""
    log "Waiting for benchmarks to complete (5-30 minutes)..."
    log "Monitor with:"
    for i in $(seq 0 $((${#INSTANCE_IPS[@]} - 1))); do
        log "  ssh -i ${KEY_FILE} ubuntu@${INSTANCE_IPS[$i]} 'tail -f /var/log/pi-benchmark.log'"
    done
    log ""

    MAX_WAIT=3600
    POLL_INTERVAL=30
    elapsed=0
    completed=0
    total=${#INSTANCE_IDS[@]}
    done_flags=()
    for i in $(seq 0 $((total - 1))); do done_flags+=(0); done

    while [ $completed -lt $total ] && [ $elapsed -lt $MAX_WAIT ]; do
        for i in $(seq 0 $((total - 1))); do
            if [ "${done_flags[$i]}" = "0" ]; then
                ip="${INSTANCE_IPS[$i]}"
                if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
                       -i "${KEY_FILE}" "ubuntu@${ip}" \
                       "test -f /home/ubuntu/benchmark_done" 2>/dev/null; then
                    log "  ✓ ${LABELS[$i]} complete!"
                    done_flags[$i]=1
                    completed=$((completed + 1))
                fi
            fi
        done
        if [ $completed -lt $total ]; then
            sleep $POLL_INTERVAL
            elapsed=$((elapsed + POLL_INTERVAL))
            log "  Waiting... ($((total - completed)) remaining, ${elapsed}s elapsed)"
        fi
    done

    # Step 4: Collect results
    log ""
    log "Collecting results..."
    for i in $(seq 0 $((total - 1))); do
        ip="${INSTANCE_IPS[$i]}"
        label="${LABELS[$i]}"
        outfile="$RESULTS_DIR/bench_${label}_$(date +%Y%m%d_%H%M%S).txt"
        scp -o StrictHostKeyChecking=no -o BatchMode=yes \
            -i "${KEY_FILE}" \
            "ubuntu@${ip}:/var/log/pi-benchmark.log" "$outfile" 2>/dev/null || true
        log "  $label → $outfile"
    done

    # Step 5: Print summary
    log ""
    log "============================================"
    log "  BENCHMARK RESULTS — $DIGITS digits"
    log "============================================"
    for i in $(seq 0 $((total - 1))); do
        label="${LABELS[$i]}"
        outfile=$(ls -t "$RESULTS_DIR"/bench_${label}_*.txt 2>/dev/null | head -1)
        if [ -n "$outfile" ] && [ -f "$outfile" ]; then
            log ""
            log "--- $label ($(echo "${CONFIGS[$i]}" | cut -d: -f2)) ---"
            grep -E "(Architecture|CPU|Cores|RAM|GMP|Total:|Binary splitting:|Final computation:|String conversion:)" "$outfile" | head -20
        fi
    done
    log ""
    log "Full results in: $RESULTS_DIR/"
fi

if [ "$DRY_RUN" = "true" ]; then
    log ""
    log "=== DRY RUN COMPLETE ==="
    log "Would launch ${#CONFIGS[@]} instances:"
    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r label itype ami_arch <<< "$config"
        log "  $label: $itype ($ami_arch)"
    done
    log ""
    log "Estimated cost: ~\$2-4 total"
    log "Run without --dry-run to execute."
fi

log ""
log "Done! Instances will be terminated automatically."
