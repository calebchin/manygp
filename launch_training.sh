#!/bin/bash
#SBATCH --job-name=patch-1p5B-test
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# bash -c "DEBUG=1 /w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/run_patching.sh qwen-1p5B"
# bash -c "/w/20252/wjcai/causal_inf/causal-mediation-analysis-rfh/run_patching.sh qwen-1p5B"

source /w/20252/wjcai/uq/manygp/venv/bin/activate
# Prefer positional argument from sbatch invocation, then env var, then default.
# SUPCON_LOSS_WEIGHT="${1:-${SUPCON_LOSS_WEIGHT:-0.0}}"

# echo "Running with supcon-loss-weight=${SUPCON_LOSS_WEIGHT}"
# python /w/20252/wjcai/uq/manygp/experiments/cifar10_supcon_sngp.py \
#   --config /w/20252/wjcai/uq/manygp/configs/cifar10_supcon_sngp.yaml \
#   --supcon-loss-weight "${SUPCON_LOSS_WEIGHT}"

python experiments/cifar10_sngp.py --config configs/cifar10_sngp.yaml