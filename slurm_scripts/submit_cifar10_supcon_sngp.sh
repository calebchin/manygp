#!/bin/bash
#SBATCH --job-name=cifar10_supcon_sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

# Usage:
#   sbatch submit_cifar10_supcon_sngp.sh [SEED] [RUN_NAME]
# Example:
#   sbatch submit_cifar10_supcon_sngp.sh 0 supcon_sngp_seed0

REPO=/w/20252/davida/manygp/manygp
SEED=${1:-""}
RUN_NAME=${2:-""}

mkdir -p /w/20252/davida/manygp_logs
cd $REPO

source /w/20252/davida/venv/bin/activate

export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running CIFAR-10 SupCon+SNGP | seed=${SEED:-none} | run=${RUN_NAME:-default}"

python -u experiments/cifar10_supcon_sngp.py --config configs/cifar10_supcon_sngp.yaml \
    ${SEED:+--seed "$SEED"} \
    ${RUN_NAME:+--run-name "$RUN_NAME"}
