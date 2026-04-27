#!/bin/bash
#SBATCH --job-name=cifar100
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch submit_cifar100.sh <experiment>
#
# Available experiments:
#   classifier       - Vanilla supervised classifier (cross-entropy)
#   supcon           - Supervised contrastive training + k-NN eval
#   sngp             - SNGP with spectrally normalized ResNet backbone
#   supcon_sngp      - Joint supervised contrastive + SNGP training
#   frozen_sngp      - SNGP with frozen pretrained backbone (run supcon first)
#
# Example:
#   sbatch submit_cifar100.sh sngp

EXPERIMENT=${1:-""}

if [ -z "$EXPERIMENT" ]; then
    echo "Error: no experiment specified."
    echo "Usage: sbatch submit_cifar100.sh <experiment>"
    echo "Available: classifier, supcon, sngp, supcon_sngp, frozen_sngp"
    exit 1
fi

case "$EXPERIMENT" in
    classifier|supcon|sngp|supcon_sngp|frozen_sngp)
        ;;
    *)
        echo "Error: unknown experiment '$EXPERIMENT'"
        echo "Available: classifier, supcon, sngp, supcon_sngp, frozen_sngp"
        exit 1
        ;;
esac

REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running cifar100_${EXPERIMENT} experiment..."

python -u experiments/cifar100_${EXPERIMENT}.py --config configs/cifar100_${EXPERIMENT}.yaml
