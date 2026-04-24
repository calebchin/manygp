from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gmm_eval import evaluate_checkpoint_path, summarize_metric_rows, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SNGP checkpoints with GMM posterior and OOD metrics.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint path. Pass multiple times for multiple checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        help="Optional text file with one checkpoint path per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for embeddings and CSV outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="sngp_gmm_metrics",
        help="Prefix for generated CSV files.",
    )
    parser.add_argument(
        "--fallback-config",
        type=str,
        default=None,
        help="Fallback YAML config if the checkpoint does not contain config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, for example 'cpu' or 'cuda'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Recompute embeddings even if cached embeddings already exist.",
    )
    parser.add_argument(
        "--covariance-reg",
        type=float,
        default=1e-3,
        help="Relative covariance regularization for label-assigned and EM GMMs.",
    )
    parser.add_argument(
        "--em-max-iter",
        type=int,
        default=100,
        help="Maximum EM iterations for the unsupervised GMM.",
    )
    parser.add_argument(
        "--em-tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for the unsupervised GMM.",
    )
    parser.add_argument(
        "--em-random-state",
        type=int,
        default=42,
        help="Random seed for the unsupervised GMM.",
    )
    return parser.parse_args()


def load_checkpoint_paths(direct_paths: list[str], checkpoint_file: str | None) -> list[Path]:
    checkpoint_paths: list[str] = list(direct_paths)
    if checkpoint_file is not None:
        file_path = Path(checkpoint_file)
        with file_path.open() as handle:
            for line in handle:
                candidate = line.strip()
                if not candidate or candidate.startswith("#"):
                    continue
                checkpoint_paths.append(candidate)

    resolved_paths: list[Path] = []
    seen: set[Path] = set()
    for checkpoint_path in checkpoint_paths:
        resolved = Path(checkpoint_path).expanduser().resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        seen.add(resolved)
        resolved_paths.append(resolved)

    if not resolved_paths:
        raise ValueError("Provide at least one --checkpoint or --checkpoint-file.")
    return resolved_paths


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_evaluations(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    checkpoint_paths = load_checkpoint_paths(args.checkpoint, args.checkpoint_file)
    output_dir = Path(args.output_dir).resolve()
    device = resolve_device(args.device)

    metric_rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoint_paths:
        row, _ = evaluate_checkpoint_path(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            fallback_config_path=args.fallback_config,
            device=device,
            force_embeddings=args.force_embeddings,
            num_workers_override=args.num_workers,
            covariance_reg=args.covariance_reg,
            em_max_iter=args.em_max_iter,
            em_tol=args.em_tol,
            em_random_state=args.em_random_state,
        )
        metric_rows.append(row)

    summary_rows = summarize_metric_rows(metric_rows)
    return metric_rows, summary_rows


def write_outputs(
    metric_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    per_checkpoint_path = output_dir / f"{output_prefix}_per_checkpoint.csv"
    summary_path = output_dir / f"{output_prefix}_summary.csv"
    write_csv_rows(metric_rows, per_checkpoint_path)
    write_csv_rows(summary_rows, summary_path)
    return per_checkpoint_path, summary_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    metric_rows, summary_rows = run_evaluations(args)
    per_checkpoint_path, summary_path = write_outputs(
        metric_rows=metric_rows,
        summary_rows=summary_rows,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"Wrote per-checkpoint metrics to {per_checkpoint_path}")
    print(f"Wrote summary metrics to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
