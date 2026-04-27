#!/bin/bash
#SBATCH --job-name=cifar10_dinov2_mlp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch slurm_scripts/submit_april20_cifar10_dinov2_mlp.sh [seed]
#
# To run multiple seeds:
#   for seed in 42 43 44 45 46; do sbatch slurm_scripts/submit_april20_cifar10_dinov2_mlp.sh $seed; done

REPO=/w/20251/cchin/manygp/manygp
SEED=${1:-42}

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running CIFAR-10 DINOv2 frozen-backbone plain MLP baseline (seed=${SEED})..."

python -u experiments/cifar10_dinov2_mlp.py \
    --config configs/experiment_april20_cifar10_dinov2_mlp.yaml \
    --seed $SEED \
    --run-name "cifar10_dinov2_mlp_vits14_seed${SEED}"
