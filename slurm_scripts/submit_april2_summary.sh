#!/bin/bash
#SBATCH --job-name=apr2_summary
#SBATCH --partition=gpunodes
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/w/20252/davida/manygp_logs/%j_summary.out
#SBATCH --error=/w/20252/davida/manygp_logs/%j_summary.err

REPO=/w/20252/davida/manygp/manygp

cd "$REPO"
source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)

echo "============================================"
echo "  april_4_experiments — generating summary"
echo "============================================"

python -u experiments/summarize_april2.py \
    --project april_4_experiments \
    --entity sta414manygp
