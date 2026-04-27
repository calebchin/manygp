#!/bin/bash
#SBATCH --job-name=cifar10-supcon-sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-17%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

source /w/20252/wjcai/uq/manygp/venv/bin/activate
cd /w/20252/wjcai/uq/manygp || exit 1

WEIGHTS=(0 0.02 0.5 4 32 256)
SEEDS=(42 43 44)

num_weights=${#WEIGHTS[@]}
num_seeds=${#SEEDS[@]}
total_jobs=$((num_weights * num_seeds))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}

if (( task_id < 0 || task_id >= total_jobs )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..$((total_jobs - 1))"
  exit 1
fi

weight_index=$((task_id / num_seeds))
seed_index=$((task_id % num_seeds))

weight="${WEIGHTS[weight_index]}"
seed="${SEEDS[seed_index]}"
run_name="lambda_exp_${weight}_${seed}"

CMD=(
  python3
  experiments/cifar10_supcon_sngp.py
  --config
  configs/cifar10_supcon_sngp.yaml
  --supcon-loss-weight
  "${weight}"
  --run-name
  "${run_name}"
  --seed
  "${seed}"
)

echo "Running task_id=${task_id} weight=${weight} seed=${seed} run_name=${run_name}"
"${CMD[@]}"
