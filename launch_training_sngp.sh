#!/bin/bash
#SBATCH --job-name=cifar10-sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /w/20252/wjcai/uq/manygp/venv/bin/activate
cd /w/20252/wjcai/uq/manygp || exit 1

# Prefer positional argument from sbatch invocation, then env var, then config default.
TRAIN_DATASET="${1:-${TRAIN_DATASET:-}}"
USE_SUPCON_AUGMENTATIONS="${USE_SUPCON_AUGMENTATIONS:-}"

CMD=(python3 experiments/cifar10_sngp.py --config configs/cifar10_sngp.yaml)
if [[ -n "${USE_SUPCON_AUGMENTATIONS}" ]]; then
  CMD+=(--use-supcon-augmentations "${USE_SUPCON_AUGMENTATIONS}")
fi
if [[ -n "${TRAIN_DATASET}" ]]; then
  CMD+=(--train-dataset "${TRAIN_DATASET}")
fi

echo "Running cifar10_sngp with train_dataset=${TRAIN_DATASET:-config}"
"${CMD[@]}"
