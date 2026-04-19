#!/bin/bash
#SBATCH --job-name=apr4_finalize
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX_4090|RTX_A6000"
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/w/20252/davida/manygp_logs/finalize_%j.out
#SBATCH --error=/w/20252/davida/manygp_logs/finalize_%j.err
#
# Runs after all hybrid_sngp_rff4096_sc75 seeds finish:
#   1. Rebuilds the april_4_experiments results table
#   2. Updates ambiguous_images_manifest.yaml with the two new RFF4096 model
#      checkpoints (hybrid_rff4096 seed0 and hybrid_rff4096_sc75 seed0),
#      then reruns the ambiguous images experiment (manygp_ambiguous project).

REPO=/w/20252/davida/manygp/manygp
cd "$REPO"
source /w/20252/davida/venv/bin/activate
export WANDB_API_KEY=$(cat /w/20252/davida/.wandb_api_key)

echo "=== Step 1: Rebuild results table ==="
python -u experiments/build_wandb_table.py \
    --project april_4_experiments \
    --entity  sta414manygp

echo ""
echo "=== Step 2: Update ambiguous images manifest & rerun ==="

python3 - <<'PYEOF'
import glob, yaml, sys
from pathlib import Path

manifest_path = "configs/ambiguous_images_manifest.yaml"
with open(manifest_path) as f:
    manifest = yaml.safe_load(f)

existing_names = {m["name"] for m in manifest["models"]}

def find_best(pattern):
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None

new_models = [
    {
        "name":       "Hybrid SNGP RFF4096",
        "experiment": "hybrid_sngp_rff4096",
        "config":     "configs/experiment_april4_hybrid_sngp_rff4096.yaml",
        "checkpoint": find_best("checkpoints_april4/hybrid_sngp_rff4096/seed0/**/best_model.pt")
                      or find_best("checkpoints_april4/hybrid_sngp_rff4096/seed0/best_model.pt"),
        "num_inducing_override": 4096,
    },
    {
        "name":       "Hybrid SNGP RFF4096 SC75",
        "experiment": "hybrid_sngp_rff4096",   # same training script / model class
        "config":     "configs/experiment_april4_hybrid_sngp_rff4096_sc75.yaml",
        "checkpoint": find_best("checkpoints_april4/hybrid_sngp_rff4096_sc75/seed0/**/best_model.pt")
                      or find_best("checkpoints_april4/hybrid_sngp_rff4096_sc75/seed0/best_model.pt"),
        "num_inducing_override": 4096,
    },
]

added = []
for m in new_models:
    if m["name"] in existing_names:
        print(f"  already present: {m['name']}")
        continue
    if m["checkpoint"] is None:
        print(f"  WARNING: no checkpoint found for {m['name']} — skipping")
        continue
    manifest["models"].append(m)
    added.append(m["name"])
    print(f"  added: {m['name']}  ->  {m['checkpoint']}")

if added:
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    print(f"  Manifest updated with {len(added)} new model(s).")
else:
    print("  No new models added.")
PYEOF

echo ""
echo "=== Step 3: Rerun ambiguous images experiment ==="
python -u experiments/ambiguous_images.py \
    --manifest  configs/ambiguous_images_manifest.yaml \
    --run-name  ambiguous_comparison_final \
    --num-mc    50 \
    --wandb-project manygp_ambiguous \
    --wandb-entity  sta414manygp

echo "=== All done ==="
