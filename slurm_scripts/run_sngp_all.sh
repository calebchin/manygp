#!/bin/bash
#
# Orchestration script for SNGP (vanilla).
#
# Submits the CIFAR-10 SNGP training job, then chains the OOD evaluation
# (SVHN + CIFAR-10-C corruptions) to run automatically after training finishes.
#
# Usage:
#   bash slurm_scripts/run_sngp_all.sh
#
# Results are logged to W&B project Updated_run under entity sta414manygp.
# The OOD eval config (configs/cifar10_ood_eval.yaml) points to:
#   ./checkpoints_sngp/best_model.pt   (written by the training script)

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS=$REPO/slurm_scripts

# Submit training — capture the job ID
JID=$(sbatch --parsable "$SCRIPTS/submit_cifar10_sngp.sh")
echo "Submitted SNGP training job: $JID"

# Chain OOD eval to run only after training succeeds
sbatch --dependency=afterok:"$JID" "$SCRIPTS/submit_ood_eval.sh" cifar10
echo "OOD eval (cifar10) chained after job $JID"
