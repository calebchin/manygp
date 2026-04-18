#!/bin/bash
#
# Orchestration script for april_4_experiments.
#
# Submits all training jobs across 4 experiment types × 5 seeds:
#   1. SNGP          (WRN-28-10 + spectral norm + GP, CE, RFF=1024)
#   2. SupCon+SNGP   (WRN-28-10, SupCon aug + GP, SupCon+CE)
#   3. SNGP+Aug      (WRN-28-10 + spectral norm + GP, CE, strong aug, RFF=1024)
#   4. SNGP+Aug RFF  (same as 3 but RFF=NUM_INDUCING, default 4096)
#
# After every individual seed job (success OR failure), a W&B results table
# is rebuilt and uploaded so it fills up incrementally.
#
# After all 20 jobs finish, submits 5 deterministic (Deep Ensemble member) jobs.
# After all 5 deterministic jobs finish, submits the Deep Ensemble eval job,
# followed by a final table rebuild.
#
# All training jobs use retry_lib.sh — up to 3 attempts per seed, with
# automatic W&B run cleanup between retries.
#
# Usage (from repo root):
#   bash slurm_scripts/run_april2_all.sh
#
# Override seeds or RFF count:
#   SEEDS="0 1" NUM_INDUCING=8192 bash slurm_scripts/run_april2_all.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_INDUCING="${NUM_INDUCING:-4096}"   # RFF count for experiment 4

echo "============================================"
echo "  april_4_experiments — launching all jobs"
echo "  Seeds: ${SEEDS}"
echo "  RFF experiment num_inducing: ${NUM_INDUCING}"
echo "============================================"

ALL_TRAIN_JIDS=()

# Helper: submit a table-rebuild job after a given JID (afterany = runs even if failed)
submit_table_after() {
    local PARENT_JID=$1
    sbatch --parsable \
        --dependency="afterany:${PARENT_JID}" \
        --job-name="apr4_table_snap" \
        "$SCRIPTS/submit_april4_table.sh" > /dev/null
}

for SEED in $SEEDS; do
    # ── Exp 1: SNGP ────────────────────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_sngp.sh" \
        "$SEED" "sngp_seed${SEED}")
    echo "  [seed ${SEED}] SNGP                  → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")
    submit_table_after "$JID"

    # ── Exp 2: SupCon+SNGP ─────────────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_supcon_sngp.sh" \
        "$SEED" "supcon_sngp_seed${SEED}")
    echo "  [seed ${SEED}] SupCon+SNGP            → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")
    submit_table_after "$JID"

    # ── Exp 3: SNGP+Aug (RFF=1024) ─────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_sngp_augmented.sh" \
        "$SEED" "sngp_aug_seed${SEED}")
    echo "  [seed ${SEED}] SNGP+Aug               → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")
    submit_table_after "$JID"

    # ── Exp 4: SNGP+Aug RFF (RFF=NUM_INDUCING) ─────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april4_sngp_aug_rff.sh" \
        "$SEED" "sngp_aug_rff${NUM_INDUCING}_seed${SEED}" "$NUM_INDUCING")
    echo "  [seed ${SEED}] SNGP+Aug RFF${NUM_INDUCING}        → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")
    submit_table_after "$JID"
done

echo ""
echo "  Submitted ${#ALL_TRAIN_JIDS[@]} training jobs + table rebuild after each."

# ── Deep Ensemble: 5 deterministic members (after all 20 training jobs) ───────
DEP_ALL=$(IFS=:; echo "${ALL_TRAIN_JIDS[*]}")

echo ""
echo "  Submitting 5 deterministic (Deep Ensemble member) jobs..."
DET_JIDS=()
for SEED in $SEEDS; do
    JID=$(sbatch --parsable \
        --dependency="afterok:${DEP_ALL}" \
        "$SCRIPTS/submit_april4_deterministic.sh" \
        "$SEED" "deterministic_seed${SEED}")
    echo "  [seed ${SEED}] Deterministic          → job ${JID}  (dep: afterok all ${#ALL_TRAIN_JIDS[@]})"
    DET_JIDS+=("$JID")
    submit_table_after "$JID"
done

# ── Deep Ensemble evaluation ───────────────────────────────────────────────────
DEP_DET=$(IFS=:; echo "${DET_JIDS[*]}")
ENSEMBLE_JID=$(sbatch --parsable \
    --dependency="afterok:${DEP_DET}" \
    "$SCRIPTS/submit_april4_deep_ensemble_eval.sh")
echo ""
echo "  Deep Ensemble eval         → job ${ENSEMBLE_JID}  (dep: afterok all 5 det. jobs)"

    # Table already rebuilds after the ensemble eval via the afterany hook above
    submit_table_after "$ENSEMBLE_JID"

echo ""
echo "============================================"
echo "  W&B project: april_4_experiments (entity: sta414manygp)"
echo "  Pipeline:"
echo "    ${#ALL_TRAIN_JIDS[@]} training jobs (4 exps × 5 seeds, RFF=${NUM_INDUCING} for exp 4)"
echo "    + W&B table rebuild after every seed"
echo "    → 5 deterministic jobs (Deep Ensemble members)"
echo "    → Deep Ensemble eval (job ${ENSEMBLE_JID})"
echo "    → Final table (job ${FINAL_TABLE_JID})"
echo ""
echo "  To change RFF count:  NUM_INDUCING=8192 bash slurm_scripts/run_april2_all.sh"
echo "  To run fewer seeds:   SEEDS='0 1' bash slurm_scripts/run_april2_all.sh"
echo "============================================"
