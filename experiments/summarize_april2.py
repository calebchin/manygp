"""
Summarize april_4_experiments results from W&B.

Pulls all runs in the april_4_experiments project, groups them by experiment
type (sngp / supcon_sngp / sngp_augmented), and prints a mean ± std table
over the 5 seeds, matching the format of Table 2 in the SNGP paper.

Usage:
    python experiments/summarize_april2.py
    python experiments/summarize_april2.py --project april_4_experiments --entity sta414manygp
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


EXPERIMENTS = [
    ("SNGP",             "sngp_seed"),
    ("SupCon+SNGP",      "supcon_sngp_seed"),
    ("SNGP+Aug",         "sngp_aug_seed"),
    ("SNGP+Aug RFF4096", "sngp_aug_rff4096_seed"),
    ("MS-SNGP",          "ms_sngp_seed"),
    ("MS-SNGP+CNN",      "ms_sngp_cnn_seed"),
    ("MS-SNGP+SN",       "ms_sngp_sn_seed"),
    ("MS-SNGP NoSkip",   "ms_sngp_no_skip_seed"),
    ("Hybrid SNGP",      "hybrid_sngp_seed"),
]

# Single-run experiments (no seed averaging — just show the one value)
SINGLE_RUNS = [
    ("Deep Ensemble", "deep_ensemble"),
]

# (display_name, wandb_key, scale, fmt, higher_is_better)
METRICS = [
    ("Clean Acc (%)",     "test/accuracy",           100.0, ".2f", True),
    ("Corrupt Acc (%)",   "test/corrupted_accuracy", 100.0, ".2f", True),
    ("Clean ECE",         "test/ece",                  1.0, ".4f", False),
    ("Corrupt ECE",       "test/corrupted_ece",         1.0, ".4f", False),
    ("Clean NLL",         "test/nll",                  1.0, ".4f", False),
    ("AUPR SVHN",         "ood/svhn/ds_aupr",           1.0, ".4f", True),
    ("AUPR CIFAR-100",    "ood/cifar100/ds_aupr",       1.0, ".4f", True),
]


def fetch_runs(project: str, entity: str) -> list:
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=200)
    return list(runs)


def collect_single(runs: list, exact_name: str) -> dict[str, list[float]]:
    """Return {metric_key: [value]} for the run whose name exactly matches exact_name."""
    collected: dict[str, list[float]] = {m[1]: [] for m in METRICS}
    for run in runs:
        if run.name != exact_name:
            continue
        summary = run.summary._json_dict
        for _, key, *_ in METRICS:
            val = summary.get(key)
            if val is not None:
                collected[key].append(float(val))
        break  # only use the first match
    return collected


def collect_summary(runs: list, name_prefix: str) -> dict[str, list[float]]:
    """Return {metric_key: [values across seeds]} for runs matching name_prefix."""
    collected: dict[str, list[float]] = {m[1]: [] for m in METRICS}
    matched = 0
    for run in runs:
        if not run.name.startswith(name_prefix):
            continue
        matched += 1
        summary = run.summary._json_dict
        for _, key, *_ in METRICS:
            val = summary.get(key)
            if val is not None:
                collected[key].append(float(val))
    if matched == 0:
        print(f"  WARNING: no runs found with prefix '{name_prefix}'", file=sys.stderr)
    return collected


def fmt_cell(values: list[float], scale: float, fmt: str) -> str:
    if not values:
        zero = format(0.0, fmt)
        return f"{zero} ± {zero} (0 seeds)"
    arr = np.array(values) * scale
    mean, std = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else 0.0
    return f"{mean:{fmt}} ± {std:{fmt}} ({len(values)} seeds)"


def print_table(results: dict[str, dict[str, list[float]]]) -> None:
    col_width = 28
    metric_names = [m[0] for m in METRICS]
    exp_names   = [e[0] for e in EXPERIMENTS]

    header = f"{'Method':<16}" + "".join(f"{m:>{col_width}}" for m in metric_names)
    sep    = "-" * len(header)

    print()
    print("=" * len(header))
    print("  april_4_experiments — Summary (mean ± std over seeds)")
    print("=" * len(header))
    print(header)
    print(sep)

    for exp_name, (_, prefix) in zip(exp_names, EXPERIMENTS):
        data = results[prefix]
        row = f"{exp_name:<16}"
        for _, key, scale, fmt, _ in METRICS:
            cell = fmt_cell(data.get(key, []), scale, fmt)
            row += f"{cell:>{col_width}}"
        print(row)

    print(sep)

    # Single-run rows (Deep Ensemble etc.)
    for exp_name, exact_name in SINGLE_RUNS:
        data = results.get(f"_single_{exact_name}", {})
        row = f"{exp_name:<16}"
        for _, key, scale, fmt, _ in METRICS:
            vals = data.get(key, [])
            if vals:
                cell = format(vals[0] * scale, fmt)
            else:
                cell = format(0.0, fmt) + " (pending)"
            row += f"{cell:>{col_width}}"
        print(row)

    print("=" * len(header))
    print()

    # Seed counts
    print("Seed counts:")
    for _, prefix in EXPERIMENTS:
        data = results[prefix]
        counts = {k: len(v) for k, v in data.items() if v}
        n = max(counts.values()) if counts else 0
        print(f"  {prefix}: {n} seeds")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="april_4_experiments")
    parser.add_argument("--entity",  default="sta414manygp")
    args = parser.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project} ...")
    runs = fetch_runs(args.project, args.entity)
    print(f"  Found {len(runs)} total runs.")

    results: dict[str, dict[str, list[float]]] = {}
    for _, prefix in EXPERIMENTS:
        results[prefix] = collect_summary(runs, prefix)
    for _, exact_name in SINGLE_RUNS:
        results[f"_single_{exact_name}"] = collect_single(runs, exact_name)

    print_table(results)


if __name__ == "__main__":
    main()
