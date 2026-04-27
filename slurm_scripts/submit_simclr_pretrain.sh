#!/bin/bash
#SBATCH --job-name=cifar10_simclr_pretrain
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch submit_simclr_pretrain.sh
# Pretrain wide resnet 28-10 on cifar-10 with self-supervised simclr approach
REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running CIFAR-10 simclr pretraining"

python -u experiments/cifar10_simclr_pretrain.py \
    --config configs/cifar10_simclr_pretrain.yaml
