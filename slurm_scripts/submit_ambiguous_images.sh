#!/bin/bash
#SBATCH --job-name=ambiguous_imgs
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/w/20252/davida/manygp_logs/ambiguous_images_%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/ambiguous_images_%j.err
#
# Run the ambiguous-image uncertainty comparison across all 8 SNGP checkpoints.
# Logs bar charts + images to W&B project: manygp_ambiguous
#
# Usage:
#   sbatch submit_ambiguous_images.sh [RUN_NAME]

REPO=/w/20252/davida/manygp/manygp
RUN_NAME=${1:-"ambiguous_comparison_seed0"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running ambiguous images experiment | run=${RUN_NAME}"

python -u experiments/ambiguous_images.py \
    --manifest  configs/ambiguous_images_manifest.yaml \
    --run-name  "$RUN_NAME" \
    --num-mc    50 \
    --wandb-project manygp_ambiguous \
    --wandb-entity  sta414manygp
