#!/bin/bash
#
# Orchestration script for SupCon+SNGP (custom joint model).
#
# Submits the CIFAR-10 SupCon+SNGP training job, then chains OOD evaluation
# to run automatically after training finishes.
#
# Usage:
#   bash slurm_scripts/run_supcon_sngp_all.sh
#
# After training, update configs/cifar10_ood_eval.yaml checkpoint_path to:
#   ./checkpoints_supcon_sngp/<timestamp>/best_model.pt
# (or copy best_model.pt from the timestamped dir to ./checkpoints_supcon_sngp/best_model.pt)

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS=$REPO/slurm_scripts

# Submit training
JID=$(sbatch --parsable "$SCRIPTS/submit_cifar10_supcon_sngp.sh")
echo "Submitted SupCon+SNGP training job: $JID"

# Chain OOD eval
sbatch --dependency=afterok:"$JID" "$SCRIPTS/submit_ood_eval.sh" cifar10
echo "OOD eval (cifar10) chained after job $JID"
