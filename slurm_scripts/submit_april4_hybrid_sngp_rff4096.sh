#!/bin/bash
#SBATCH --job-name=apr4_hyb_rff
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090"
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err
#
# Hybrid SNGP with 4096 RFF features.
# RTX_4090 only (A6000 is ~2× slower; 4096 precision matrix inversion is already
# the bottleneck — don't make it worse).
# 48h wall time: 4096 pinv is 64× more expensive than 1024; precision_inv caching
# is applied in src/models/sngp.py so eval is fast, but training forward passes
# (building Λ_c) still process 4096-dim outer products each batch.
#
# Usage:
#   sbatch submit_april4_hybrid_sngp_rff4096.sh <SEED> [RUN_NAME]

REPO=/w/20252/davida/manygp/manygp
SEED=${1:?"SEED argument required"}
RUN_NAME=${2:-"hybrid_sngp_rff4096_seed${SEED}"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running Hybrid SNGP RFF4096 | seed=${SEED} | run=${RUN_NAME}"

source "$REPO/slurm_scripts/retry_lib.sh"

run_with_retry 3 sta414manygp april_4_experiments "$RUN_NAME" \
    python -u experiments/cifar10_hybrid_sngp.py \
        --config configs/experiment_april4_hybrid_sngp_rff4096.yaml \
        --seed "$SEED" \
        --run-name "$RUN_NAME"
