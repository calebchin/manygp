#!/bin/bash
#
# Orchestration script for Deep Ensemble.
#
# Submits the CIFAR-10 Deep Ensemble job.  OOD evaluation (SVHN, CIFAR-100)
# is performed inline at the end of the training script — no separate OOD
# eval job is needed.
#
# Usage:
#   bash slurm_scripts/run_deep_ensemble_all.sh
#
# Results (train/val/test accuracy, NLL, ECE, OOD AUPR) are logged to W&B
# project Updated_run under entity sta414manygp.

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS=$REPO/slurm_scripts

JID=$(sbatch --parsable "$SCRIPTS/submit_cifar10_deep_ensemble.sh")
echo "Submitted Deep Ensemble job: $JID  (OOD eval is inline — no separate job needed)"
