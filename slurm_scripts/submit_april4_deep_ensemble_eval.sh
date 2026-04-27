#!/bin/bash
#SBATCH --job-name=apr4_ensemble
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j_ensemble.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j_ensemble.err

REPO=/w/20252/davida/manygp/manygp
CKPT_DIR="$REPO/checkpoints_april4/deterministic"

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "============================================"
echo "  april_4 — Deep Ensemble evaluation"
echo "  Loading from: ${CKPT_DIR}/seed*/best_model.pt"
echo "============================================"

python -u experiments/cifar10_deep_ensemble_eval.py \
    --config configs/experiment_april4_deterministic.yaml \
    --checkpoints "${CKPT_DIR}/seed0/best_model.pt" \
                  "${CKPT_DIR}/seed1/best_model.pt" \
                  "${CKPT_DIR}/seed2/best_model.pt" \
                  "${CKPT_DIR}/seed3/best_model.pt" \
                  "${CKPT_DIR}/seed4/best_model.pt" \
    --run-name "deep_ensemble"
