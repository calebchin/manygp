#!/bin/bash
#SBATCH --job-name=rff4096_eval
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

# Usage:
#   sbatch submit_rff4096_eval_only.sh <SEED> [NUM_INDUCING]
#
# Loads best_model.pt for the given seed, resumes the existing crashed W&B run,
# and logs test / OOD / CIFAR-C metrics into it.

REPO=/w/20252/davida/manygp/manygp
SEED=${1:?"SEED argument required"}
NUM_INDUCING=${2:-4096}
RUN_NAME="sngp_aug_rff${NUM_INDUCING}_seed${SEED}"

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Eval-only pass | seed=${SEED} | num_inducing=${NUM_INDUCING} | run=${RUN_NAME}"

python -u experiments/eval_only_sngp_aug.py \
    --config configs/experiment_april2_sngp_augmented.yaml \
    --seed "$SEED" \
    --num-inducing "$NUM_INDUCING" \
    --run-name "$RUN_NAME"
