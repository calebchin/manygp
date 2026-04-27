#!/bin/bash
#SBATCH --job-name=cifar10_frozen_backbone_sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch slurm_scripts/submit_april18_cifar10_frozen_backbone_sngp.sh
#
# Runs CIFAR-10 frozen-backbone SNGP with OOD evaluation (april 18 experiment).

REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running CIFAR-10 frozen-backbone SNGP..."

python -u experiments/cifar10_frozen_backbone_sngp.py \
    --config configs/experiment_april18_cifar10_frozen_backbone_sngp.yaml
