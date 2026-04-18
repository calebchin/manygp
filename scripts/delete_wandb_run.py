#!/usr/bin/env python3
"""
Delete all W&B runs with a given display name from a project.

Usage:
    python scripts/delete_wandb_run.py <entity> <project> <run_name>

Deletes every run whose display_name (the human-readable name shown in W&B)
matches run_name exactly. This is called automatically by retry_lib.sh before
re-submitting a failed training job, so partial/crashed runs don't pollute
the project.
"""
import sys
import os

import wandb

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <entity> <project> <run_name>", file=sys.stderr)
        sys.exit(1)

    entity, project, run_name = sys.argv[1], sys.argv[2], sys.argv[3]

    # Suppress wandb output noise
    os.environ.setdefault("WANDB_SILENT", "true")

    api = wandb.Api()
    path = f"{entity}/{project}"

    try:
        runs = api.runs(path, filters={"display_name": run_name}, per_page=50)
        deleted = 0
        for run in runs:
            print(f"  Deleting run '{run.name}' (id={run.id}, state={run.state})")
            run.delete()
            deleted += 1
        if deleted == 0:
            print(f"  No runs named '{run_name}' found in {path} — nothing to delete.")
        else:
            print(f"  Deleted {deleted} run(s) named '{run_name}' from {path}.")
    except Exception as e:
        # Never let cleanup failure block a retry
        print(f"  WARNING: W&B cleanup failed ({e}). Continuing anyway.", file=sys.stderr)

if __name__ == "__main__":
    main()
