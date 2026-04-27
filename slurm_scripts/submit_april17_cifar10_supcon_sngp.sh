#!/bin/bash
#SBATCH --job-name=cifar10_supcon_sngp_mml
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch submit_april17_cifar10_supcon_sngp.sh
#
# Runs CIFAR-10 SupCon+SNGP with MML length scale optimization (april 17 experiment).

REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running CIFAR-10 SupCon+SNGP with MML length scale optimization..."

python -u experiments/cifar10_supcon_sngp.py \
    --config configs/experiment_april17_cifar10_supcon_sngp.yaml
