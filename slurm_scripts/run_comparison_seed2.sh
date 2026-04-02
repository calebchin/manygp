#!/bin/bash
#
# Compare SNGP vs SupCon+SNGP — seed 2
#
# Submits both jobs simultaneously so they run in parallel.
# All metrics (train/val/test accuracy, NLL, ECE) are logged to:
#   W&B entity:  sta414manygp
#   W&B project: Updated_run
#   Run names:   sngp_seed2  /  supcon_sngp_seed2
#
# Usage:
#   bash slurm_scripts/run_comparison_seed2.sh

set -euo pipefail

SEED=2
REPO=/w/20252/davida/manygp/manygp
SCRIPTS=$REPO/slurm_scripts

echo "=== Submitting SNGP vs SupCon+SNGP comparison — seed ${SEED} ==="

JID_SNGP=$(sbatch --parsable "$SCRIPTS/submit_cifar10_sngp.sh" $SEED "sngp_seed${SEED}")
echo "  SNGP job:        $JID_SNGP  (run: sngp_seed${SEED})"

JID_SUPCON=$(sbatch --parsable "$SCRIPTS/submit_cifar10_supcon_sngp.sh" $SEED "supcon_sngp_seed${SEED}")
echo "  SupCon+SNGP job: $JID_SUPCON  (run: supcon_sngp_seed${SEED})"

echo ""
echo "Both jobs running in parallel. Monitor with:"
echo "  squeue -j ${JID_SNGP},${JID_SUPCON}"
echo "  tail -f $REPO/logs/${JID_SNGP}.out"
echo "  tail -f $REPO/logs/${JID_SUPCON}.out"
