"""
Build and upload a results summary table to W&B for april_4_experiments.

Fetches all finished/running runs via the W&B API (no CSV export needed),
groups them by experiment type, computes mean ± std over seeds, renders a
matplotlib table, and logs the image back to the same W&B project as a
dedicated "results_table" run.

Designed to be called after every seed finishes so the table updates
incrementally. Runs that are still in progress are skipped; only runs whose
state is "finished" are included.

Usage:
    python experiments/build_wandb_table.py
    python experiments/build_wandb_table.py --project april_4_experiments --entity sta414manygp
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

os.environ.setdefault("WANDB_SILENT", "true")

# ── Metric definitions ────────────────────────────────────────────────────────
# (display_name, wandb_summary_key, multiply_by_100_for_display)
METRICS: list[tuple[str, str, bool]] = [
    ("CIFAR-10\nAccuracy (%)",       "test/accuracy",           True),
    ("CIFAR-10\nECE",                "test/ece",                False),
    ("CIFAR-10\nNLL",                "test/nll",                False),
    ("CIFAR-10-C\nAccuracy (%)",     "test/corrupted_accuracy", True),
    ("CIFAR-10-C\nECE",              "test/corrupted_ece",      False),
    ("SVHN (OOD)\nAUPR (%)",         "ood/svhn/ds_aupr",        True),
    ("CIFAR-100 (OOD)\nAUPR (%)",    "ood/cifar100/ds_aupr",    True),
    ("SVHN (OOD)\nmp AUPR (%)",      "ood/svhn/mp_aupr",        True),
    ("CIFAR-100 (OOD)\nmp AUPR (%)", "ood/cifar100/mp_aupr",    True),
]

# Map experiment.name → pretty display name.
# Any name not in this dict is shown as-is (handles future experiments automatically).
DISPLAY_NAMES: dict[str, str] = {
    "cifar10_sngp":             "SNGP",
    "cifar10_sngp_augmented":   "SNGP+Aug",
    "cifar10_sngp_aug_rff":     "SNGP+Aug RFF4096",
    "cifar10_supcon_sngp":      "SupCon+SNGP",
    "cifar10_ms_sngp":          "MS-SNGP",
    "cifar10_ms_sngp_cnn":      "MS-SNGP+CNN",
    "cifar10_ms_sngp_sn":       "MS-SNGP+SN",
    "cifar10_ms_sngp_no_skip":  "MS-SNGP NoSkip",
    "cifar10_hybrid_sngp":      "Hybrid SNGP",
    "cifar10_classifier":       "Deterministic",   # individual member — filtered out below
    "deep_ensemble":            "Deep Ensemble",
}
# Experiment names that are individual ensemble members and should be excluded
# from the table (their ensemble result appears as "deep_ensemble" instead)
EXCLUDE_EXP_NAMES = {"cifar10_classifier", "results_table"}

# Desired row order (any experiment not listed here appears at the bottom)
ROW_ORDER = [
    "SNGP",
    "SNGP+Aug",
    "SNGP+Aug RFF4096",
    "SupCon+SNGP",
    "MS-SNGP",
    "MS-SNGP+CNN",
    "MS-SNGP+SN",
    "MS-SNGP NoSkip",
    "Hybrid SNGP",
    "Deep Ensemble",
]


# ── Data fetching ─────────────────────────────────────────────────────────────

_RUN_PREFIX_MAP: list[tuple[str, str]] = [
    # (run name prefix, canonical experiment name)
    ("sngp_aug_rff",      "cifar10_sngp_aug_rff"),   # must come before sngp_aug_
    ("sngp_aug_",         "cifar10_sngp_augmented"),
    ("supcon_sngp_",      "cifar10_supcon_sngp"),
    ("ms_sngp_cnn_",      "cifar10_ms_sngp_cnn"),
    ("ms_sngp_sn_",       "cifar10_ms_sngp_sn"),
    ("ms_sngp_no_skip_", "cifar10_ms_sngp_no_skip"),
    ("hybrid_sngp_",     "cifar10_hybrid_sngp"),
    ("ms_sngp_",          "cifar10_ms_sngp"),
    ("sngp_seed",         "cifar10_sngp"),
    ("deep_ensemble",     "deep_ensemble"),
    ("det_seed",          "cifar10_classifier"),
]


def _exp_name_from_run(run_name: str) -> str:
    """Derive canonical experiment name from run display name."""
    for prefix, canonical in _RUN_PREFIX_MAP:
        if run_name.startswith(prefix):
            return canonical
    return run_name  # fallback: unknown experiment, show raw name


def fetch_finished_runs(entity: str, project: str) -> pd.DataFrame:
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=300)

    rows = []
    for run in runs:
        # Skip jobs still actively running or queued — include finished/crashed/failed
        if run.state in ("running", "pending"):
            continue

        exp_name = _exp_name_from_run(run.name)

        if exp_name in EXCLUDE_EXP_NAMES:
            continue

        row: dict = {"_exp_name": exp_name, "_run_name": run.name, "_state": run.state}
        summary = run.summary._json_dict
        for _, key, _ in METRICS:
            row[key] = summary.get(key, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["_exp_name", "_run_name"] + [k for _, k, _ in METRICS]
    )


# ── Table building ────────────────────────────────────────────────────────────

def build_formatted_table(df: pd.DataFrame) -> pd.DataFrame:
    """Group by experiment, compute mean ± std, return formatted string DataFrame."""
    if df.empty:
        return pd.DataFrame()

    metric_keys = [k for _, k, _ in METRICS]
    display_labels = [d for d, _, _ in METRICS]

    grouped = df.groupby("_exp_name")[metric_keys].agg(["mean", "std", "count"])

    formatted = {}
    for (disp, key, pct) in METRICS:
        col_mean = (key, "mean")
        col_std  = (key, "std")
        col_n    = (key, "count")
        cells = []
        for exp in grouped.index:
            m = grouped.loc[exp, col_mean]
            s = grouped.loc[exp, col_std]
            n = int(grouped.loc[exp, col_n])
            if pd.isna(m):
                cells.append("N/A")
            else:
                scale = 100.0 if pct else 1.0
                m_s, s_s = m * scale, (s * scale if not pd.isna(s) else 0.0)
                cells.append(f"{m_s:.2f} ± {s_s:.2f}\n({n} seeds)")
        formatted[disp] = cells

    result = pd.DataFrame(formatted, index=grouped.index)
    result.index = [DISPLAY_NAMES.get(n, n) for n in result.index]
    result.index.name = "Method"

    # Sort rows according to ROW_ORDER
    ordered = [r for r in ROW_ORDER if r in result.index]
    rest    = [r for r in result.index if r not in ROW_ORDER]
    return result.loc[ordered + rest]


def render_table_image(table: pd.DataFrame, n_runs: int) -> bytes:
    """Render the DataFrame as a matplotlib table PNG, returned as bytes."""
    nrows = len(table)
    ncols = len(table.columns)
    fig_w = max(16, ncols * 2.2)
    fig_h = max(2.0, nrows * 0.9 + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("tight")
    ax.axis("off")

    ax.set_title(
        f"april_4_experiments  —  Results Table  (based on {n_runs} completed runs)",
        fontsize=11, fontweight="bold", pad=12,
    )

    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        rowLabels=table.index,
        loc="center",
        cellLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(9)
    mpl_table.scale(1.0, 3.2)

    for (row, col), cell in mpl_table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eaf6")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── W&B upload ────────────────────────────────────────────────────────────────

def upload_table(
    entity: str,
    project: str,
    table: pd.DataFrame,
    img_bytes: bytes,
    n_runs: int,
) -> None:
    import wandb

    # Use a stable run ID so each invocation updates the same run
    stable_id = f"results-table-{project}"[:64]

    run = wandb.init(
        project=project,
        entity=entity,
        id=stable_id,
        name="results_table",
        resume="allow",
        tags=["results_table"],
    )

    # Log the image (wandb.Image needs a PIL image, not raw bytes)
    pil_img = PILImage.open(io.BytesIO(img_bytes))
    img = wandb.Image(pil_img, caption=f"Results table ({n_runs} finished runs)")
    run.log({"results/table_image": img, "results/n_finished_runs": n_runs})

    # Also log as a native W&B Table for interactive filtering
    wb_table = wandb.Table(
        columns=["Method"] + list(table.columns),
        data=[[idx] + list(row) for idx, row in zip(table.index, table.values)],
    )
    run.log({"results/summary_table": wb_table})

    run.finish()
    print(f"  Uploaded results table to {entity}/{project} (run id: {stable_id})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and upload results table to W&B")
    parser.add_argument("--project", default="april_4_experiments")
    parser.add_argument("--entity",  default="sta414manygp")
    parser.add_argument("--save-png", default=None,
                        help="Also save PNG to this local path (optional)")
    args = parser.parse_args()

    print(f"Fetching finished runs from {args.entity}/{args.project} ...")
    df = fetch_finished_runs(args.entity, args.project)

    if df.empty:
        print("  No finished runs yet — nothing to plot.")
        return

    n_runs = len(df)
    unique_exps = df["_exp_name"].unique().tolist()
    print(f"  {n_runs} finished runs across experiments: {unique_exps}")

    table = build_formatted_table(df)
    print(table.to_string())

    img_bytes = render_table_image(table, n_runs)

    if args.save_png:
        Path(args.save_png).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_png).write_bytes(img_bytes)
        print(f"  PNG saved to {args.save_png}")

    upload_table(args.entity, args.project, table, img_bytes, n_runs)


if __name__ == "__main__":
    main()
