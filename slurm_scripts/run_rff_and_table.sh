#!/bin/bash
#
# Submits ONLY the 5 RFF seeds + wires up the W&B table to rebuild after each.
# Run this when the rest of april_4_experiments is already in progress/done.
#
# Usage (from repo root):
#   bash slurm_scripts/run_rff_and_table.sh
#
# Override:
#   SEEDS="0 1" NUM_INDUCING=8192 bash slurm_scripts/run_rff_and_table.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_INDUCING="${NUM_INDUCING:-4096}"

echo "============================================"
echo "  april_4_experiments — new experiments"
echo "  Seeds: ${SEEDS}"
echo "  RFF num_inducing: ${NUM_INDUCING}"
echo "============================================"

# Helper — submit a table rebuild after a job (always fires, regardless of success/fail)
table_after() {
    sbatch --parsable \
        --dependency="afterany:$1" \
        --job-name="apr4_table" \
        "$SCRIPTS/submit_april4_table.sh" > /dev/null
}

echo ""
echo "  ── SNGP+Aug RFF${NUM_INDUCING} ────────────────────────────────"
for SEED in $SEEDS; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_sngp_aug_rff.sh" \
        "$SEED" "sngp_aug_rff${NUM_INDUCING}_seed${SEED}" "$NUM_INDUCING")
    echo "  [seed ${SEED}] SNGP+Aug RFF${NUM_INDUCING} → job ${JID}"
    table_after "$JID"
done

echo ""
echo "  ── MS Loss + SNGP ──────────────────────────────────────────"
for SEED in $SEEDS; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_ms_sngp.sh" \
        "$SEED" "ms_sngp_seed${SEED}")
    echo "  [seed ${SEED}] MS-SNGP               → job ${JID}"
    table_after "$JID"
done

echo ""
echo "============================================"
