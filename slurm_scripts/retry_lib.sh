#!/bin/bash
# retry_lib.sh — source this file to get run_with_retry()
#
# Usage in a submit script:
#   source "$REPO/slurm_scripts/retry_lib.sh"
#
#   run_with_retry <max_retries> <entity> <project> <run_name> <python ...args...>
#
# Behaviour:
#   1. Runs the python command.
#   2. On failure: deletes any W&B run whose display_name == run_name in
#      entity/project (avoids ghost partial runs), then retries.
#   3. Repeats up to max_retries times total.
#   4. Exits 0 on success, 1 if all attempts exhausted.

run_with_retry() {
    local MAX_RETRIES=$1
    local ENTITY=$2
    local PROJECT=$3
    local RUN_NAME=$4
    shift 4
    local CMD=("$@")

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        echo ""
        echo "━━━ Attempt ${attempt}/${MAX_RETRIES} | run='${RUN_NAME}' ━━━"
        if "${CMD[@]}"; then
            echo "━━━ SUCCESS on attempt ${attempt} ━━━"
            return 0
        fi

        echo "━━━ FAILED on attempt ${attempt} — cleaning up W&B run '${RUN_NAME}' ━━━"
        python -u "$(dirname "${BASH_SOURCE[0]}")/../scripts/delete_wandb_run.py" \
            "$ENTITY" "$PROJECT" "$RUN_NAME" || true

        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            echo "Retrying in 30s..."
            sleep 30
        fi
    done

    echo "━━━ All ${MAX_RETRIES} attempts failed for run '${RUN_NAME}' ━━━"
    return 1
}
