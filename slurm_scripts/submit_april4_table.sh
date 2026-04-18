#!/bin/bash
#SBATCH --job-name=apr4_table
#SBATCH --partition=gpunodes
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j_table.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j_table.err

REPO=/w/20252/davida/manygp/manygp
PNG_DIR=/w/20252/davida/manygp_logs/tables

mkdir -p "$PNG_DIR"
cd "$REPO"

source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)

echo "============================================"
echo "  april_4_experiments — building results table"
echo "============================================"

python -u experiments/build_wandb_table.py \
    --project april_4_experiments \
    --entity  sta414manygp \
    --save-png "${PNG_DIR}/results_table_$(date +%Y%m%d_%H%M%S).png"
