#!/bin/bash
#SBATCH --job-name=ood_eval
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

# Usage:
#   sbatch submit_ood_eval.sh <dataset>
#
# Available datasets:
#   cifar10   - Evaluate CIFAR-10 model (OOD: SVHN, CIFAR-100; corruptions: CIFAR-10-C)
#   cifar100  - Evaluate CIFAR-100 model (OOD: SVHN, CIFAR-10; corruptions: CIFAR-100-C)
#
# Example:
#   sbatch submit_ood_eval.sh cifar10
#
# Note: Update checkpoint_path in configs/<dataset>_ood_eval.yaml before submitting.

DATASET=${1:-""}

if [ -z "$DATASET" ]; then
    echo "Error: no dataset specified."
    echo "Usage: sbatch submit_ood_eval.sh <dataset>"
    echo "Available: cifar10, cifar100"
    exit 1
fi

case "$DATASET" in
    cifar10|cifar100)
        ;;
    *)
        echo "Error: unknown dataset '$DATASET'"
        echo "Available: cifar10, cifar100"
        exit 1
        ;;
esac

REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

echo "Running ${DATASET} OOD evaluation..."

python -u experiments/${DATASET}_ood_eval.py --config configs/${DATASET}_ood_eval.yaml
