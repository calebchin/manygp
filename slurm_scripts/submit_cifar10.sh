#!/bin/bash
#SBATCH --job-name=cifar10_dspp
#SBATCH --partition=gpunodes
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/w/20251/cchin/manygp/manygp/logs/%j.out
#SBATCH --error=/w/20251/cchin/manygp/manygp/logs/%j.err

REPO=/w/20251/cchin/manygp/manygp

cd $REPO
mkdir -p $REPO/logs

source $REPO/.venv/bin/activate

RUN_NAME=${1:-""}  # pass as first argument to sbatch, e.g. sbatch submit_cifar10.sh my-run

python -u experiments/cifar10_dspp.py --config configs/cifar10_dspp.yaml \
    ${RUN_NAME:+--run-name "$RUN_NAME"}
