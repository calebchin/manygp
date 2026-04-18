#!/bin/bash
#SBATCH --job-name=apr4_cifarc
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err
#
# Eval-only pass for runs that finished training (best_model.pt exists) but were
# killed before OOD / CIFAR-C evaluation completed.
# Resumes the existing W&B run and logs all remaining metrics.
#
# Usage:
#   sbatch submit_cifarc_only_april4.sh <EXP> <SEED> [RUN_NAME]
#
#   EXP  = hybrid_sngp | ms_sngp | ms_sngp_no_skip | ms_sngp_sn
#
# Examples:
#   sbatch submit_cifarc_only_april4.sh hybrid_sngp    0 hybrid_sngp_seed0
#   sbatch submit_cifarc_only_april4.sh hybrid_sngp    2 hybrid_sngp_seed2
#   sbatch submit_cifarc_only_april4.sh ms_sngp        1 ms_sngp_seed1
#   sbatch submit_cifarc_only_april4.sh ms_sngp_no_skip 3 ms_sngp_no_skip_seed3

REPO=/w/20252/davida/manygp/manygp
EXP=${1:?"EXP argument required (hybrid_sngp | ms_sngp | ms_sngp_no_skip | ms_sngp_sn)"}
SEED=${2:?"SEED argument required"}
RUN_NAME=${3:-"${EXP}_seed${SEED}"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Eval-only ${EXP} | seed=${SEED} | run=${RUN_NAME}"

case "$EXP" in
    hybrid_sngp)
        CONFIG="configs/experiment_april4_hybrid_sngp.yaml"
        ;;
    ms_sngp_no_skip)
        CONFIG="configs/experiment_april4_ms_sngp_no_skip.yaml"
        ;;
    ms_sngp_sn)
        CONFIG="configs/experiment_april4_ms_sngp_sn.yaml"
        ;;
    ms_sngp)
        CONFIG="configs/experiment_april4_ms_sngp.yaml"
        ;;
    *)
        echo "Unknown experiment: ${EXP}"
        exit 1
        ;;
esac

python -u experiments/eval_only_april4.py \
    --experiment "$EXP" \
    --config     "$CONFIG" \
    --seed       "$SEED" \
    --run-name   "$RUN_NAME"
