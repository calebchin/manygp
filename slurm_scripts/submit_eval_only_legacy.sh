#!/bin/bash
#SBATCH --job-name=apr4_eval_leg
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err
#
# Eval-only pass for april_2 experiments (sngp, sngp_augmented, sngp_aug_rff4096)
# that completed training but crashed before OOD/CIFAR-C finished.
# Uses experiments/eval_only_sngp_aug.py which handles SNGPResNetClassifier.
#
# Usage:
#   sbatch submit_eval_only_legacy.sh <EXP_TYPE> <SEED> [RUN_NAME] [NUM_INDUCING]
#
#   EXP_TYPE = sngp | sngp_augmented | sngp_aug_rff4096
#
# Examples:
#   sbatch submit_eval_only_legacy.sh sngp           1 sngp_seed1
#   sbatch submit_eval_only_legacy.sh sngp_augmented 3 sngp_aug_seed3
#   sbatch submit_eval_only_legacy.sh sngp_aug_rff4096 1 sngp_aug_rff4096_seed1 4096

REPO=/w/20252/davida/manygp/manygp
EXP_TYPE=${1:?"EXP_TYPE required (sngp | sngp_augmented | sngp_aug_rff4096)"}
SEED=${2:?"SEED required"}
RUN_NAME=${3:-"${EXP_TYPE}_seed${SEED}"}
NUM_INDUCING=${4:-""}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Eval-only ${EXP_TYPE} | seed=${SEED} | run=${RUN_NAME}"

# Resolve config and checkpoint path
case "$EXP_TYPE" in
    sngp)
        CONFIG="configs/experiment_april2_sngp.yaml"
        CKPT="${REPO}/checkpoints_april2/sngp/seed${SEED}/best_model.pt"
        ;;
    sngp_augmented)
        CONFIG="configs/experiment_april2_sngp_augmented.yaml"
        CKPT="${REPO}/checkpoints_april2/sngp_augmented/seed${SEED}/best_model.pt"
        ;;
    sngp_aug_rff4096)
        CONFIG="configs/experiment_april2_sngp_augmented.yaml"
        CKPT="${REPO}/checkpoints_april2/sngp_augmented/sngp_aug_rff4096/seed${SEED}/best_model.pt"
        ;;
    *)
        echo "Unknown EXP_TYPE: ${EXP_TYPE}"
        exit 1
        ;;
esac

EXTRA=""
if [ -n "$NUM_INDUCING" ]; then
    EXTRA="--num-inducing $NUM_INDUCING"
fi

python -u experiments/eval_only_sngp_aug.py \
    --config     "$CONFIG" \
    --seed       "$SEED" \
    --run-name   "$RUN_NAME" \
    --checkpoint "$CKPT" \
    $EXTRA
