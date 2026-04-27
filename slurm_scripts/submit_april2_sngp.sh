#!/bin/bash
#SBATCH --job-name=apr2_sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

# Usage:
#   sbatch submit_april2_sngp.sh <SEED> <RUN_NAME>
# Example:
#   sbatch submit_april2_sngp.sh 0 sngp_seed0

REPO=/w/20252/davida/manygp/manygp
SEED=${1:?"SEED argument required"}
RUN_NAME=${2:-"sngp_seed${SEED}"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate

export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running experiment_april_2 SNGP | seed=${SEED} | run=${RUN_NAME}"

python -u experiments/cifar10_sngp.py \
    --config configs/experiment_april2_sngp.yaml \
    --seed "$SEED" \
    --run-name "$RUN_NAME"
