#!/bin/bash
#SBATCH --job-name=cifar10-sngp
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-user=wjcai@cs.toronto.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

source /w/20252/wjcai/uq/manygp/venv/bin/activate
cd /w/20252/wjcai/uq/manygp || exit 1

TRAIN_DATASETS=(standard supcon_two_view)
SEEDS=(42 43 44 45 46)

num_train_datasets=${#TRAIN_DATASETS[@]}
num_seeds=${#SEEDS[@]}
total_jobs=$((num_train_datasets * num_seeds))
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}

if (( task_id < 0 || task_id >= total_jobs )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..$((total_jobs - 1))"
  exit 1
fi

train_dataset_index=$((task_id / num_seeds))
seed_index=$((task_id % num_seeds))

train_dataset="${TRAIN_DATASETS[train_dataset_index]}"
seed="${SEEDS[seed_index]}"
run_name="cifar10_sngp_train_dataset_${train_dataset}_seed_${seed}"

CMD=(
  python3
  experiments/cifar10_sngp.py
  --config
  configs/cifar10_sngp.yaml
  --train-dataset
  "${train_dataset}"
  --seed
  "${seed}"
  --run-name
  "${run_name}"
)

echo "Running task_id=${task_id} train_dataset=${train_dataset} seed=${seed} run_name=${run_name}"
"${CMD[@]}"
