#!/bin/bash
#
# Submit 3 new ablation experiments (5 seeds each = 15 jobs total):
#
#   ms_sngp_no_skip  — WRN-28-10 depth/width, SN, NO skip connections, MS Loss
#   ms_sngp_sn       — WRN-28-10 with SN backbone, MS Loss, cosine geometry (no layer norm)
#   hybrid_sngp      — WRN-28-10, SupCon + MS Loss + CE
#
# The W&B table rebuilds after every individual seed finishes.
#
# Usage (from repo root):
#   bash slurm_scripts/run_new_ablations.sh
#
# Override:
#   SEEDS="0 1" bash slurm_scripts/run_new_ablations.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"
SEEDS="${SEEDS:-0 1 2 3 4}"

echo "============================================"
echo "  april_4_experiments — 3 new ablations"
echo "  Seeds: ${SEEDS}"
echo "============================================"

table_after() {
    sbatch --parsable \
        --dependency="afterany:$1" \
        --job-name="apr4_table" \
        "$SCRIPTS/submit_april4_table.sh" > /dev/null
}

echo ""
echo "  ── MS-SNGP NoSkip (WRN-28-10, no residuals, SN, MS Loss) ──"
for SEED in $SEEDS; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_ms_sngp_no_skip.sh" \
        "$SEED" "ms_sngp_no_skip_seed${SEED}")
    echo "  [seed ${SEED}] MS-SNGP NoSkip → job ${JID}"
    table_after "$JID"
done

echo ""
echo "  ── MS-SNGP+SN (WRN-28-10, SN backbone, MS Loss, cosine GP) ─"
for SEED in $SEEDS; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_ms_sngp_sn.sh" \
        "$SEED" "ms_sngp_sn_seed${SEED}")
    echo "  [seed ${SEED}] MS-SNGP+SN   → job ${JID}"
    table_after "$JID"
done

echo ""
echo "  ── Hybrid SNGP (SupCon + MS Loss + CE) ─────────────────────"
for SEED in $SEEDS; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_hybrid_sngp.sh" \
        "$SEED" "hybrid_sngp_seed${SEED}")
    echo "  [seed ${SEED}] Hybrid SNGP  → job ${JID}"
    table_after "$JID"
done

echo ""
echo "============================================"
echo "  Submitted 15 jobs (5 per experiment)"
echo "  Table rebuilds after every seed."
echo "============================================"
