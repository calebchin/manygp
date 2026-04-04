#!/bin/bash
#
# Orchestration script for april_4_experiments.
#
# Submits all 15 training jobs: 5 seeds × 3 experiments (SNGP, SupCon+SNGP, SNGP+Aug).
# After each training job a snapshot summary is printed (afterany dependency).
# After all 15 jobs finish, submits 5 deterministic (Deep Ensemble member) jobs.
# After all 5 deterministic jobs finish, submits the Deep Ensemble eval job.
# A final summary runs after the ensemble eval.
#
# Usage (from the repo root):
#   bash slurm_scripts/run_april2_all.sh
#
# To run only specific seeds, set SEEDS before calling:
#   SEEDS="0 1" bash slurm_scripts/run_april2_all.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"
SEEDS="${SEEDS:-0 1 2 3 4}"

echo "============================================"
echo "  april_4_experiments — launching all jobs"
echo "  Seeds: ${SEEDS}"
echo "============================================"

ALL_TRAIN_JIDS=()

for SEED in $SEEDS; do
    # ── Experiment 1: SNGP ──────────────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_sngp.sh" \
        "$SEED" "sngp_seed${SEED}")
    echo "  [seed ${SEED}] SNGP              → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")

    # Snapshot summary after this job (runs whether job succeeded or failed)
    sbatch --parsable \
        --dependency="afterany:${JID}" \
        --job-name="apr4_summary_snap" \
        "$SCRIPTS/submit_april2_summary.sh" > /dev/null

    # ── Experiment 2: SupCon+SNGP ───────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_supcon_sngp.sh" \
        "$SEED" "supcon_sngp_seed${SEED}")
    echo "  [seed ${SEED}] SupCon+SNGP       → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")

    sbatch --parsable \
        --dependency="afterany:${JID}" \
        --job-name="apr4_summary_snap" \
        "$SCRIPTS/submit_april2_summary.sh" > /dev/null

    # ── Experiment 3: SNGP+Aug ──────────────────────────────────────────────
    JID=$(sbatch --parsable "$SCRIPTS/submit_april2_sngp_augmented.sh" \
        "$SEED" "sngp_aug_seed${SEED}")
    echo "  [seed ${SEED}] SNGP+Aug          → job ${JID}"
    ALL_TRAIN_JIDS+=("$JID")

    sbatch --parsable \
        --dependency="afterany:${JID}" \
        --job-name="apr4_summary_snap" \
        "$SCRIPTS/submit_april2_summary.sh" > /dev/null
done

echo ""
echo "  Submitted ${#ALL_TRAIN_JIDS[@]} training jobs + snapshot summary after each."

# ── Deep Ensemble: 5 deterministic members (after all 15 training jobs) ──────
#    Uses afterany so they start even if some SNGP jobs fail, since they are
#    independent. Change to afterok if you want strict ordering.
DEP_ALL=$(IFS=:; echo "${ALL_TRAIN_JIDS[*]}")

echo ""
echo "  Submitting 5 deterministic (Deep Ensemble member) jobs..."
DET_JIDS=()
for SEED in $SEEDS; do
    JID=$(sbatch --parsable \
        --dependency="afterok:${DEP_ALL}" \
        "$SCRIPTS/submit_april4_deterministic.sh" \
        "$SEED" "deterministic_seed${SEED}")
    echo "  [seed ${SEED}] Deterministic      → job ${JID}  (dep: afterok all 15)"
    DET_JIDS+=("$JID")

    # Snapshot summary after each deterministic job too
    sbatch --parsable \
        --dependency="afterany:${JID}" \
        --job-name="apr4_summary_snap" \
        "$SCRIPTS/submit_april2_summary.sh" > /dev/null
done

# ── Deep Ensemble evaluation (after all 5 deterministic jobs succeed) ─────────
DEP_DET=$(IFS=:; echo "${DET_JIDS[*]}")
ENSEMBLE_JID=$(sbatch --parsable \
    --dependency="afterok:${DEP_DET}" \
    "$SCRIPTS/submit_april4_deep_ensemble_eval.sh")
echo ""
echo "  Deep Ensemble eval → job ${ENSEMBLE_JID}  (dep: afterok all 5 det. jobs)"

# ── Final summary after ensemble eval ─────────────────────────────────────────
FINAL_JID=$(sbatch --parsable \
    --dependency="afterany:${ENSEMBLE_JID}" \
    --job-name="apr4_summary_final" \
    "$SCRIPTS/submit_april2_summary.sh")
echo "  Final summary       → job ${FINAL_JID}  (dep: afterany ensemble eval)"

echo ""
echo "============================================"
echo "  W&B project: april_4_experiments (entity: sta414manygp)"
echo "  Pipeline:"
echo "    15 training jobs (SNGP × 3 exps × 5 seeds)"
echo "    + snapshot summary after each training job"
echo "    → 5 deterministic jobs (Deep Ensemble members)"
echo "    + snapshot summary after each det. job"
echo "    → Deep Ensemble eval (job ${ENSEMBLE_JID})"
echo "    → Final summary (job ${FINAL_JID})"
echo "============================================"
