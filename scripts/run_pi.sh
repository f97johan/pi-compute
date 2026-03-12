#!/usr/bin/env bash
#
# run_pi.sh — Launch an EC2 instance to compute pi digits, stream output to S3
#
# Launches a single instance, builds the project, computes pi, and streams
# the output file + checkpoints to an S3 bucket. The instance is NOT terminated
# automatically (tagged auto-delete=no) so you can inspect results.
#
# Usage:
#   ./scripts/run_pi.sh --key-name KEY --key-file PATH --s3-bucket BUCKET \
#       --digits N --instance-type TYPE [--region REGION] [--disk-gb N]
#
# Example:
#   ./scripts/run_pi.sh \
#       --key-name pi-bench \
#       --key-file ~/.ssh/pi-bench.pem \
#       --s3-bucket my-pi-results \
#       --digits 50000000000 \
#       --instance-type r8g.24xlarge \
#       --disk-gb 200
#
# Prerequisites:
#   - AWS CLI configured
#   - EC2 key pair
#   - S3 bucket (created beforehand)
#   - IAM instance profile with S3 write access (or use --iam-profile)

set -eo pipefail

# ============================================================
# Configuration
# ============================================================

DIGITS=""
INSTANCE_TYPE=""
KEY_NAME=""
KEY_FILE=""
S3_BUCKET=""
S3_PREFIX="pi-compute"
REGION="us-west-2"
DISK_GB=200
IAM_PROFILE=""
SUBNET_ID=""
SECURITY_GROUP=""
GITHUB_REPO="https://github.com/f97johan/pi-compute.git"
DRY_RUN=false

# ============================================================
# Parse arguments
# ============================================================

while [ $# -gt 0 ]; do
    case "$1" in
        --digits)         DIGITS="$2"; shift 2 ;;
        --instance-type)  INSTANCE_TYPE="$2"; shift 2 ;;
        --key-name)       KEY_NAME="$2"; shift 2 ;;
        --key-file)       KEY_FILE="$2"; shift 2 ;;
        --s3-bucket)      S3_BUCKET="$2"; shift 2 ;;
        --s3-prefix)      S3_PREFIX="$2"; shift 2 ;;
        --region)         REGION="$2"; shift 2 ;;
        --disk-gb)        DISK_GB="$2"; shift 2 ;;
        --iam-profile)    IAM_PROFILE="$2"; shift 2 ;;
        --subnet-id)      SUBNET_ID="$2"; shift 2 ;;
        --sg)             SECURITY_GROUP="$2"; shift 2 ;;
        --repo)           GITHUB_REPO="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 --key-name KEY --key-file PATH --s3-bucket BUCKET --digits N --instance-type TYPE"
            echo ""
            echo "Required:"
            echo "  --digits N            Number of pi digits to compute"
            echo "  --instance-type TYPE  EC2 instance type (e.g., r8g.24xlarge)"
            echo "  --key-name KEY        EC2 key pair name"
            echo "  --key-file PATH       Path to .pem private key file"
            echo "  --s3-bucket BUCKET    S3 bucket for output (must exist)"
            echo ""
            echo "Optional:"
            echo "  --s3-prefix PREFIX    S3 key prefix (default: pi-compute)"
            echo "  --region REGION       AWS region (default: us-west-2)"
            echo "  --disk-gb N           EBS volume size in GB (default: 200)"
            echo "  --iam-profile NAME    IAM instance profile for S3 access"
            echo "  --subnet-id ID        Subnet ID"
            echo "  --sg SG_ID            Security group ID"
            echo "  --repo URL            Git repo URL"
            echo "  --dry-run             Print commands without executing"
            echo ""
            echo "The instance is tagged auto-delete=no and NOT terminated on exit."
            echo "Output is streamed to s3://BUCKET/PREFIX/DIGITS/"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required args
missing=""
[ -z "$DIGITS" ] && missing="$missing --digits"
[ -z "$INSTANCE_TYPE" ] && missing="$missing --instance-type"
[ -z "$S3_BUCKET" ] && missing="$missing --s3-bucket"
if [ "$DRY_RUN" = "false" ]; then
    [ -z "$KEY_NAME" ] && missing="$missing --key-name"
    [ -z "$KEY_FILE" ] && missing="$missing --key-file"
    [ -n "$KEY_FILE" ] && [ ! -f "$KEY_FILE" ] && { echo "ERROR: Key file not found: $KEY_FILE"; exit 1; }
fi
if [ -n "$missing" ]; then
    echo "ERROR: Missing required arguments:$missing"
    echo "Run with --help for usage."
    exit 1
fi

# ============================================================
# Helper functions
# ============================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Determine AMI architecture from instance type
get_ami_arch() {
    local itype="$1"
    # Graviton instances have 'g' in the family (c7g, r8g, m7g, etc.)
    case "$itype" in
        *g.*|*g[0-9]*) echo "arm64" ;;
        *)             echo "x86_64" ;;
    esac
}

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
# User data script
# ============================================================

generate_userdata() {
    cat <<USERDATA
#!/bin/bash
set -ex
exec > /var/log/pi-compute.log 2>&1

echo "=== Pi Computation Starting ==="
echo "Instance type: ${INSTANCE_TYPE}"
echo "Digits: ${DIGITS}"
echo "S3 bucket: ${S3_BUCKET}"
echo "S3 prefix: ${S3_PREFIX}"
date -u

# Install dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake libgmp-dev git awscli

# Clone and build
cd /home/ubuntu
git clone ${GITHUB_REPO} pi-compute
cd pi-compute
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j\$(nproc)

# Run tests to verify build
./build/tests/pi_tests

# Create work directories
mkdir -p /home/ubuntu/output
mkdir -p /home/ubuntu/ckpt

echo "=== Build complete, starting computation ==="
date -u

# Run pi computation
./build/src/pi_compute \\
    --digits ${DIGITS} \\
    --verbose \\
    --checkpoint /home/ubuntu/ckpt \\
    --output /home/ubuntu/output/pi_${DIGITS}.txt \\
    2>&1 | tee /home/ubuntu/output/compute.log

echo "=== Computation complete ==="
date -u

# Upload results to S3
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${DIGITS}"

echo "Uploading output to \${S3_PATH}/ ..."

# Upload the pi digits file
aws s3 cp /home/ubuntu/output/pi_${DIGITS}.txt "\${S3_PATH}/pi_${DIGITS}.txt" \\
    --no-progress || echo "WARNING: Failed to upload pi digits file"

# Upload the compute log
aws s3 cp /home/ubuntu/output/compute.log "\${S3_PATH}/compute.log" \\
    --no-progress || echo "WARNING: Failed to upload compute log"

# Upload the full system log
aws s3 cp /var/log/pi-compute.log "\${S3_PATH}/system.log" \\
    --no-progress || echo "WARNING: Failed to upload system log"

# Upload checkpoints (for potential resume)
aws s3 sync /home/ubuntu/ckpt/ "\${S3_PATH}/checkpoints/" \\
    --no-progress || echo "WARNING: Failed to upload checkpoints"

# Upload system info
{
    echo "Instance type: ${INSTANCE_TYPE}"
    echo "Architecture: \$(uname -m)"
    echo "CPU: \$(lscpu | grep 'Model name' | head -1 | sed 's/.*: *//')"
    echo "Cores: \$(nproc)"
    echo "RAM: \$(free -h | awk '/Mem:/ {print \$2}')"
    echo "Disk: \$(df -h /home/ubuntu | tail -1 | awk '{print \$2}')"
    echo "GMP: \$(dpkg -s libgmp-dev | grep Version)"
    echo "Digits: ${DIGITS}"
    echo "Completed: \$(date -u)"
} > /home/ubuntu/output/system_info.txt
aws s3 cp /home/ubuntu/output/system_info.txt "\${S3_PATH}/system_info.txt" \\
    --no-progress || echo "WARNING: Failed to upload system info"

echo "=== Upload complete ==="
echo "Results at: \${S3_PATH}/"
date -u

# Signal completion
touch /home/ubuntu/computation_done
USERDATA
}

# ============================================================
# Main
# ============================================================

AMI_ARCH=$(get_ami_arch "$INSTANCE_TYPE")
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${DIGITS}"

log "Pi Computation Launcher"
log "  Digits:    $DIGITS"
log "  Instance:  $INSTANCE_TYPE ($AMI_ARCH)"
log "  Disk:      ${DISK_GB} GB gp3"
log "  S3 output: $S3_PATH/"
log "  Region:    $REGION"
log ""

if [ "$DRY_RUN" = "true" ]; then
    log "=== DRY RUN ==="
    log "Would launch $INSTANCE_TYPE with ${DISK_GB}GB disk"
    log "Output would go to $S3_PATH/"
    log ""
    log "User data script:"
    log "---"
    generate_userdata
    log "---"
    exit 0
fi

# Look up AMI
log "Looking up Ubuntu 22.04 AMI ($AMI_ARCH)..."
AMI=$(lookup_ami "$AMI_ARCH")
log "  AMI: $AMI"

# Create security group if needed
CREATED_SG=""
if [ -z "$SECURITY_GROUP" ]; then
    SG_NAME="pi-run-ssh-$$"
    log "Creating security group: $SG_NAME..."
    VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
        --filters "Name=isDefault,Values=true" \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "")

    if [ -n "$VPC_ID" ] && [ "$VPC_ID" != "None" ]; then
        SECURITY_GROUP=$(aws ec2 create-security-group \
            --region "$REGION" \
            --group-name "$SG_NAME" \
            --description "Pi compute SSH access (auto-created)" \
            --vpc-id "$VPC_ID" \
            --query 'GroupId' --output text)
        aws ec2 authorize-security-group-ingress \
            --region "$REGION" \
            --group-id "$SECURITY_GROUP" \
            --protocol tcp --port 22 --cidr 0.0.0.0/0 \
            --output text >/dev/null
        CREATED_SG="$SECURITY_GROUP"
        log "  Security group: $SECURITY_GROUP"
    else
        log "  WARNING: No default VPC found."
    fi
fi

# Write user data to temp file
ud_tmpfile=$(mktemp)
generate_userdata > "$ud_tmpfile"

# Build block device mapping for custom disk size
BDM="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${DISK_GB},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]"

# Build optional args
subnet_arg=""
sg_arg=""
iam_arg=""
[ -n "$SUBNET_ID" ] && subnet_arg="--subnet-id $SUBNET_ID"
[ -n "$SECURITY_GROUP" ] && sg_arg="--security-group-ids $SECURITY_GROUP"
[ -n "$IAM_PROFILE" ] && iam_arg="--iam-instance-profile Name=$IAM_PROFILE"

# Launch instance
log "Launching $INSTANCE_TYPE..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --user-data "fileb://$ud_tmpfile" \
    --block-device-mappings "$BDM" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=pi-compute-${DIGITS}},{Key=Project,Value=pi-compute},{Key=auto-delete,Value=no},{Key=Digits,Value=${DIGITS}}]" \
    $subnet_arg $sg_arg $iam_arg \
    --query 'Instances[0].InstanceId' \
    --output text)

rm -f "$ud_tmpfile"

log "  Instance ID: $INSTANCE_ID"

# Wait for instance to be running
log "Waiting for instance to start..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

log "  Public IP: $PUBLIC_IP"
log ""
log "============================================"
log "  Instance launched successfully!"
log "============================================"
log ""
log "  Instance ID: $INSTANCE_ID"
log "  Public IP:   $PUBLIC_IP"
log "  Instance:    $INSTANCE_TYPE"
log "  Digits:      $DIGITS"
log "  Disk:        ${DISK_GB} GB gp3"
log ""
log "  Monitor progress:"
log "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'tail -f /var/log/pi-compute.log'"
log ""
log "  Check if done:"
log "    ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'test -f /home/ubuntu/computation_done && echo DONE || echo RUNNING'"
log ""
log "  Results will be uploaded to:"
log "    $S3_PATH/"
log ""
log "  When finished, terminate manually:"
log "    aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
if [ -n "$CREATED_SG" ]; then
    log "    aws ec2 delete-security-group --region $REGION --group-id $CREATED_SG"
fi
log ""
log "  Instance is tagged auto-delete=no — it will NOT be terminated automatically."
