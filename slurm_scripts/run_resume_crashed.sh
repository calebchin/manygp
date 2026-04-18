#!/bin/bash
#
# Submit resume / eval-only jobs for EVERY crashed run in april_4_experiments.
# Jobs are submitted in priority order so the scheduler picks them up first.
#
# Priority: hybrid_sngp → ms_sngp → ms_sngp_sn → ms_sngp_no_skip → april2 experiments
#
# Full crash inventory (verified from W&B + checkpoint directories):
#
#  ── hybrid_sngp (april_4) — checkpoints_april4/hybrid_sngp/ ─────────────────
#    seed0  → best_model.pt, crashed at CIFAR-C 11/15              → eval-only
#    seed1  → crashed training, latest ckpt epoch 185              → resume
#    seed2  → best_model.pt, crashed at CIFAR-C 8/15               → eval-only
#    seed3  → crashed training, latest ckpt epoch 185              → resume
#    seed4  → crashed training, latest ckpt epoch 180              → resume
#
#  ── ms_sngp (april_4) — checkpoints_april4/ms_sngp/ ────────────────────────
#    seed1-4→ first run completed (epoch~165-200, best_model.pt exists),
#             retry crashed from scratch — eval-only (test+OOD+CIFAR-C)
#
#  ── ms_sngp_sn (april_4) — checkpoints_april4/ms_sngp_sn/ ──────────────────
#    seed0  → COMPLETED ✓ — skipped
#    seed1-4→ crashed training, latest ckpt epoch 180-185          → resume
#
#  ── ms_sngp_no_skip (april_4) — checkpoints_april4/ms_sngp_no_skip/ ────────
#    seed0  → crashed training, latest ckpt epoch 150              → resume
#    seed1  → crashed training, latest ckpt epoch 145              → resume
#    seed2  → crashed training, latest ckpt epoch 150              → resume
#    seed3  → best_model.pt, crashed at CIFAR-C 4/15               → eval-only
#    seed4  → crashed training, latest ckpt epoch 140              → resume
#
#  ── sngp (april_2) — checkpoints_april2/sngp/ ───────────────────────────────
#    seed1  → best_model.pt exists, W&B crashed, no OOD metrics    → eval-only
#
#  ── sngp_augmented (april_2) — checkpoints_april2/sngp_augmented/ ───────────
#    seed3  → best_model.pt exists, W&B crashed, no OOD metrics    → eval-only
#
#  ── sngp_aug_rff4096 (april_2) — checkpoints_april2/sngp_augmented/sngp_aug_rff4096/ ──
#    seed0  → NO checkpoints at all                                → full re-run
#    seed1-4→ best_model.pt + test metrics exist, crashed CIFAR-C  → eval-only
#
# Usage (from repo root on cluster, after rsync from local):
#   bash slurm_scripts/run_resume_crashed.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"

echo "============================================"
echo "  april_4_experiments — resume ALL crashed"
echo "  Priority order: hybrid → ms_sngp →"
echo "                  ms_sngp_sn → ms_sngp_no_skip"
echo "                  → april2 experiments"
echo "============================================"

table_after() {
    sbatch --parsable \
        --dependency="afterany:$1" \
        --job-name="apr4_table" \
        "$SCRIPTS/submit_april4_table.sh" > /dev/null
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. hybrid_sngp  (highest priority)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 1. hybrid_sngp ───────────────────────────────────────────────────"

JID=$(sbatch --parsable "$SCRIPTS/submit_cifarc_only_april4.sh" \
    hybrid_sngp 0 hybrid_sngp_seed0)
echo "  [seed 0] eval-only → job ${JID}"
table_after "$JID"

JID=$(sbatch --parsable "$SCRIPTS/submit_resume_april4.sh" \
    hybrid_sngp 1 hybrid_sngp_seed1)
echo "  [seed 1] resume    → job ${JID}"
table_after "$JID"

JID=$(sbatch --parsable "$SCRIPTS/submit_cifarc_only_april4.sh" \
    hybrid_sngp 2 hybrid_sngp_seed2)
echo "  [seed 2] eval-only → job ${JID}"
table_after "$JID"

JID=$(sbatch --parsable "$SCRIPTS/submit_resume_april4.sh" \
    hybrid_sngp 3 hybrid_sngp_seed3)
echo "  [seed 3] resume    → job ${JID}"
table_after "$JID"

JID=$(sbatch --parsable "$SCRIPTS/submit_resume_april4.sh" \
    hybrid_sngp 4 hybrid_sngp_seed4)
echo "  [seed 4] resume    → job ${JID}"
table_after "$JID"

# ═════════════════════════════════════════════════════════════════════════════
# 2. ms_sngp
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 2. ms_sngp ───────────────────────────────────────────────────────"

for SEED in 1 2 3 4; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_cifarc_only_april4.sh" \
        ms_sngp "$SEED" "ms_sngp_seed${SEED}")
    echo "  [seed ${SEED}] eval-only → job ${JID}"
    table_after "$JID"
done

# ═════════════════════════════════════════════════════════════════════════════
# 3. ms_sngp_sn
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 3. ms_sngp_sn  (seed0 complete — skipping) ───────────────────────"

for SEED in 1 2 3 4; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_resume_april4.sh" \
        ms_sngp_sn "$SEED" "ms_sngp_sn_seed${SEED}")
    echo "  [seed ${SEED}] resume    → job ${JID}"
    table_after "$JID"
done

# ═════════════════════════════════════════════════════════════════════════════
# 4. ms_sngp_no_skip
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 4. ms_sngp_no_skip ───────────────────────────────────────────────"

for SEED in 0 1 2 4; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_resume_april4.sh" \
        ms_sngp_no_skip "$SEED" "ms_sngp_no_skip_seed${SEED}")
    echo "  [seed ${SEED}] resume    → job ${JID}"
    table_after "$JID"
done

JID=$(sbatch --parsable "$SCRIPTS/submit_cifarc_only_april4.sh" \
    ms_sngp_no_skip 3 ms_sngp_no_skip_seed3)
echo "  [seed 3] eval-only → job ${JID}"
table_after "$JID"

# ═════════════════════════════════════════════════════════════════════════════
# 5. sngp (april_2 baseline)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 5. sngp (april_2) ────────────────────────────────────────────────"

JID=$(sbatch --parsable "$SCRIPTS/submit_eval_only_legacy.sh" \
    sngp 1 sngp_seed1)
echo "  [seed 1] eval-only → job ${JID}"
table_after "$JID"

# ═════════════════════════════════════════════════════════════════════════════
# 6. sngp_augmented (april_2)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 6. sngp_augmented (april_2) ──────────────────────────────────────"

JID=$(sbatch --parsable "$SCRIPTS/submit_eval_only_legacy.sh" \
    sngp_augmented 3 sngp_aug_seed3)
echo "  [seed 3] eval-only → job ${JID}"
table_after "$JID"

# ═════════════════════════════════════════════════════════════════════════════
# 7. sngp_aug_rff4096 (april_2)
# ═════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 7. sngp_aug_rff4096 (april_2) ────────────────────────────────────"

# seed0: no checkpoints exist anywhere → full re-run from scratch
JID=$(sbatch --parsable "$SCRIPTS/submit_april4_sngp_aug_rff.sh" \
    0 sngp_aug_rff4096_seed0 4096)
echo "  [seed 0] FULL RE-RUN → job ${JID}"
table_after "$JID"

# seeds 1-4: best_model.pt + test metrics exist, only CIFAR-C needs finishing
for SEED in 1 2 3 4; do
    JID=$(sbatch --parsable "$SCRIPTS/submit_eval_only_legacy.sh" \
        sngp_aug_rff4096 "$SEED" "sngp_aug_rff4096_seed${SEED}" 4096)
    echo "  [seed ${SEED}] eval-only → job ${JID}"
    table_after "$JID"
done

echo ""
echo "============================================"
echo "  Submitted 25 jobs total:"
echo "    hybrid_sngp:      2 eval-only + 3 resume"
echo "    ms_sngp:          4 eval-only"
echo "    ms_sngp_sn:       4 resume"
echo "    ms_sngp_no_skip:  4 resume + 1 eval-only"
echo "    sngp:             1 eval-only"
echo "    sngp_augmented:   1 eval-only"
echo "    sngp_aug_rff4096: 1 re-run + 4 eval-only"
echo "  Table rebuilds after every job completes."
echo "============================================"
