#!/bin/bash
#SBATCH --job-name=apr4_ms_cnn
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

# Usage:
#   sbatch submit_april4_ms_sngp_cnn.sh <SEED> [RUN_NAME]

REPO=/w/20252/davida/manygp/manygp
SEED=${1:?"SEED argument required"}
RUN_NAME=${2:-"ms_sngp_cnn_seed${SEED}"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running MS-SNGP+CNN | seed=${SEED} | run=${RUN_NAME}"

source "$REPO/slurm_scripts/retry_lib.sh"

run_with_retry 3 sta414manygp april_4_experiments "$RUN_NAME" \
    python -u experiments/cifar10_ms_sngp_cnn.py \
        --config configs/experiment_april4_ms_sngp_cnn.yaml \
        --seed "$SEED" \
        --run-name "$RUN_NAME"
