#!/bin/bash
#SBATCH --job-name=apr4_resume
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err
#
# Resume a crashed april_4 experiment from its latest checkpoint.
# The Python script auto-detects the newest epoch checkpoint in the seed dir
# and looks up the original W&B run by display name.
#
# Usage:
#   sbatch submit_resume_april4.sh <EXP> <SEED> [RUN_NAME]
#
#   EXP  = hybrid_sngp | ms_sngp_no_skip | ms_sngp_sn
#
# Examples:
#   sbatch submit_resume_april4.sh hybrid_sngp    1 hybrid_sngp_seed1
#   sbatch submit_resume_april4.sh ms_sngp_no_skip 0 ms_sngp_no_skip_seed0
#   sbatch submit_resume_april4.sh ms_sngp_sn     1 ms_sngp_sn_seed1

REPO=/w/20252/davida/manygp/manygp
EXP=${1:?"EXP argument required (hybrid_sngp | ms_sngp_no_skip | ms_sngp_sn)"}
SEED=${2:?"SEED argument required"}
RUN_NAME=${3:-"${EXP}_seed${SEED}"}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Resuming ${EXP} | seed=${SEED} | run=${RUN_NAME}"

# Map experiment name to config + script
case "$EXP" in
    hybrid_sngp)
        CONFIG="configs/experiment_april4_hybrid_sngp.yaml"
        SCRIPT="experiments/cifar10_hybrid_sngp.py"
        ;;
    ms_sngp_no_skip)
        CONFIG="configs/experiment_april4_ms_sngp_no_skip.yaml"
        SCRIPT="experiments/cifar10_ms_sngp_no_skip.py"
        ;;
    ms_sngp_sn)
        CONFIG="configs/experiment_april4_ms_sngp_sn.yaml"
        SCRIPT="experiments/cifar10_ms_sngp_sn.py"
        ;;
    *)
        echo "Unknown experiment: ${EXP}"
        exit 1
        ;;
esac

python -u "$SCRIPT" \
    --config   "$CONFIG" \
    --seed     "$SEED" \
    --run-name "$RUN_NAME" \
    --auto-resume
