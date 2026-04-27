#!/bin/bash
#SBATCH --job-name=cifar10-supcon
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

source /w/20252/wjcai/uq/manygp/venv/bin/activate
cd /w/20252/wjcai/uq/manygp || exit 1

PROJECTION_SETTINGS=(true false)
SEEDS=(42 43 44)

num_projection_settings=${#PROJECTION_SETTINGS[@]}
num_seeds=${#SEEDS[@]}
total_jobs=$((num_projection_settings * num_seeds))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}

if (( task_id < 0 || task_id >= total_jobs )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..$((total_jobs - 1))"
  exit 1
fi

projection_index=$((task_id / num_seeds))
seed_index=$((task_id % num_seeds))

use_projection_head="${PROJECTION_SETTINGS[projection_index]}"
seed="${SEEDS[seed_index]}"

if [[ "${use_projection_head}" == "true" ]]; then
  projection_flag="--use-projection-head"
  projection_label="with_projection"
else
  projection_flag="--no-use-projection-head"
  projection_label="no_projection"
fi

run_name="cifar10_supcon_v2_${projection_label}_seed_${seed}"

CMD=(
  python3
  experiments/cifar10_supcon.py
  --config
  configs/cifar10_supcon.yaml
  "${projection_flag}"
  --run-name
  "${run_name}"
  --seed
  "${seed}"
)

echo "Running task_id=${task_id} use_projection_head=${use_projection_head} seed=${seed} run_name=${run_name}"
"${CMD[@]}"
