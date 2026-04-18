#!/bin/bash
#SBATCH --job-name=apr4_rff
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j.err

# Usage:
#   sbatch submit_april4_sngp_aug_rff.sh <SEED> <RUN_NAME> [NUM_INDUCING]
#
# NUM_INDUCING controls the number of random Fourier features in the GP head.
# More features → better kernel approximation → better calibration/OOD,
# at the cost of a larger GP weight matrix and more GPU memory.
# Default: 4096  (config default is 1024; this is the whole point of this run)

REPO=/w/20252/davida/manygp/manygp
SEED=${1:?"SEED argument required"}
RUN_NAME=${2:-"sngp_aug_rff4096_seed${SEED}"}
NUM_INDUCING=${3:-4096}

mkdir -p /w/20252/davida/manygp_logs
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)
export CUDA_LAUNCH_BLOCKING=1

echo "Running SNGP+Aug RFF | seed=${SEED} | run=${RUN_NAME} | num_inducing=${NUM_INDUCING}"

source "$REPO/slurm_scripts/retry_lib.sh"

run_with_retry 3 sta414manygp april_4_experiments "$RUN_NAME" \
    python -u experiments/cifar10_sngp_augmented.py \
        --config configs/experiment_april2_sngp_augmented.yaml \
        --seed "$SEED" \
        --run-name "$RUN_NAME" \
        --num-inducing "$NUM_INDUCING"
