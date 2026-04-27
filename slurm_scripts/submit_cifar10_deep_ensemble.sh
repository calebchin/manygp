#!/bin/bash
#SBATCH --job-name=cifar10_deep_ensemble
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

REPO=/w/20252/davida/manygp/manygp

mkdir -p /w/20252/davida/manygp_logs
cd $REPO

source /w/20252/davida/venv/bin/activate

echo "Running CIFAR-10 Deep Ensemble experiment (5 members)..."

python -u experiments/cifar10_deep_ensemble.py --config configs/cifar10_deep_ensemble.yaml
