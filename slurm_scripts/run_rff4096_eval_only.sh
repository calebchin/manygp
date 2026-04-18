#!/bin/bash
#
# Submit eval-only jobs for all 5 crashed sngp_aug_rff4096 seeds.
#
# Each job resumes its existing W&B run and logs test + OOD + CIFAR-C metrics
# into the same run (no new runs created).
#
# Usage (from repo root):
#   bash slurm_scripts/run_rff4096_eval_only.sh
#
# Override seeds or num_inducing:
#   SEEDS="0 2" NUM_INDUCING=4096 bash slurm_scripts/run_rff4096_eval_only.sh

set -euo pipefail

REPO=/w/20252/davida/manygp/manygp
SCRIPTS="$REPO/slurm_scripts"
SEEDS="${SEEDS:-0 1 2 3 4}"
NUM_INDUCING="${NUM_INDUCING:-4096}"

echo "============================================"
echo "  sngp_aug_rff${NUM_INDUCING} — eval-only pass"
echo "  Seeds: ${SEEDS}"
echo "============================================"
echo ""

for SEED in $SEEDS; do
    RUN_NAME="sngp_aug_rff${NUM_INDUCING}_seed${SEED}"
    CKPT="/w/20252/davida/manygp/manygp/checkpoints_april2/sngp_augmented/sngp_aug_rff${NUM_INDUCING}/seed${SEED}/best_model.pt"

    if [ ! -f "$CKPT" ]; then
        echo "  [seed ${SEED}] WARNING: checkpoint not found at ${CKPT} — skipping"
        continue
    fi

    JID=$(sbatch --parsable "$SCRIPTS/submit_rff4096_eval_only.sh" "$SEED" "$NUM_INDUCING")
    echo "  [seed ${SEED}] ${RUN_NAME} → job ${JID}"

    # Rebuild the W&B table after this eval job completes
    sbatch --parsable \
        --dependency="afterany:${JID}" \
        --job-name="apr4_table" \
        "$SCRIPTS/submit_april4_table.sh" > /dev/null
done

echo ""
echo "============================================"
echo "  Submitted eval-only jobs for seeds: ${SEEDS}"
echo "  Table will rebuild after each job."
echo "============================================"
