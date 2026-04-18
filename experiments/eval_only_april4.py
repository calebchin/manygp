"""
Eval-only pass for april_4 experiments that completed training (best_model.pt
exists) but were killed by the SLURM time limit before OOD / CIFAR-C finished.

Supports three experiment types that share the same GP-SNGP eval code:
  • hybrid_sngp      — SNGPResNetClassifier  (SN WRN-28-10, SupCon+MS+CE)
  • ms_sngp_no_skip  — WRNNoSkipSupConSNGPClassifier (SN WRN-28-10 no skips, MS+CE)
  • ms_sngp_sn       — SNGPResNetClassifier  (SN WRN-28-10, MS+CE)

Loads best_model.pt, resumes the existing W&B run (same run, not a new one),
runs test + OOD + CIFAR-C, and logs all metrics back to that run.

Usage:
    python experiments/eval_only_april4.py \\
        --experiment hybrid_sngp \\
        --config     configs/experiment_april4_hybrid_sngp.yaml \\
        --seed       0 \\
        --run-name   hybrid_sngp_seed0

    # Or point directly at the best_model.pt:
    python experiments/eval_only_april4.py \\
        --experiment hybrid_sngp \\
        --config     configs/experiment_april4_hybrid_sngp.yaml \\
        --run-name   hybrid_sngp_seed0 \\
        --checkpoint /w/20252/davida/manygp/manygp/checkpoints_april4/hybrid_sngp/seed0/20250411_123456/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.sngp import laplace_predictive_probs
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper


# ── Experiment registry ────────────────────────────────────────────────────────

_EXPERIMENT_REGISTRY = {
    # ── april_4 ablations ──────────────────────────────────────────────────────
    "hybrid_sngp": {
        "model_class": "SNGPResNetClassifier",
        "ckpt_subdir":  "hybrid_sngp",
    },
    "ms_sngp_no_skip": {
        "model_class": "WRNNoSkipSupConSNGPClassifier",
        "ckpt_subdir":  "ms_sngp_no_skip",
    },
    "ms_sngp_sn": {
        "model_class": "SNGPResNetClassifier",
        "ckpt_subdir":  "ms_sngp_sn",
    },
    # ── ms_sngp (plain backbone, no SN) ───────────────────────────────────────
    "ms_sngp": {
        "model_class": "CifarResNetSupConSNGPClassifier",
        "ckpt_subdir":  "ms_sngp",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_wandb_run_id(entity: str, project: str, run_name: str) -> "str | None":
    try:
        import wandb as _wandb
        api = _wandb.Api()
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_name},
            per_page=10,
        )
        for r in runs:
            if r.name == run_name:
                print(f"  Found W&B run '{run_name}' → id={r.id}  state={r.state}")
                return r.id
    except Exception as exc:
        print(f"  W&B run lookup failed: {exc}")
    return None


def find_best_model(seed_dir: Path) -> "Path | None":
    """Find best_model.pt inside any timestamped subdirectory of seed_dir."""
    # Multiple timestamps might exist (e.g. from failed earlier runs); take latest
    candidates = sorted(seed_dir.glob("**/best_model.pt"), reverse=True)
    if candidates:
        return candidates[0]
    # Fallback: pick the top-k checkpoint with highest accuracy from its name
    import re
    best_acc, best_path = -1.0, None
    for p in seed_dir.glob("**/*_epoch*_accuracy*.pt"):
        m = re.search(r"_accuracy([0-9.]+)\.pt$", p.name)
        if m:
            acc = float(m.group(1))
            if acc > best_acc:
                best_acc, best_path = acc, p
    return best_path


def build_model(model_class_name: str, model_cfg: dict, device: torch.device):
    from src.models.sngp import SNGPResNetClassifier, WRNNoSkipSupConSNGPClassifier
    from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier

    if model_class_name == "CifarResNetSupConSNGPClassifier":
        # Plain (no SN) WRN-28-10 backbone used by ms_sngp
        return CifarResNetSupConSNGPClassifier(
            embedding_dim   = model_cfg["embedding_dim"],
            num_classes     = model_cfg["num_classes"],
            widen_factor    = model_cfg.get("widen_factor", 10),
            hidden_dims     = model_cfg["hidden_dims"],
            dropout_rate    = model_cfg["dropout_rate"],
            num_inducing    = model_cfg["num_inducing"],
            ridge_penalty   = model_cfg["ridge_penalty"],
            feature_scale   = model_cfg["feature_scale"],
            gp_cov_momentum = model_cfg["gp_cov_momentum"],
            normalize_input = model_cfg["normalize_input"],
            kernel_type     = model_cfg.get("kernel_type", "legacy"),
            input_normalization = model_cfg.get("input_normalization", None),
            kernel_scale    = model_cfg.get("kernel_scale", 1.0),
            length_scale    = model_cfg.get("length_scale", 1.0),
        ).to(device)

    # SN WRN-28-10 variants (SNGPResNetClassifier, WRNNoSkipSupConSNGPClassifier)
    klass = {
        "SNGPResNetClassifier":          SNGPResNetClassifier,
        "WRNNoSkipSupConSNGPClassifier":  WRNNoSkipSupConSNGPClassifier,
    }[model_class_name]
    return klass(
        num_classes     = model_cfg["num_classes"],
        widen_factor    = model_cfg.get("widen_factor", 10),
        hidden_dim      = model_cfg["hidden_dim"],
        spec_norm_bound = model_cfg["spec_norm_bound"],
        num_inducing    = model_cfg["num_inducing"],
        ridge_penalty   = model_cfg["ridge_penalty"],
        feature_scale   = model_cfg["feature_scale"],
        gp_cov_momentum = model_cfg["gp_cov_momentum"],
        normalize_input = model_cfg["normalize_input"],
        kernel_type     = model_cfg.get("kernel_type", "legacy"),
        input_normalization = model_cfg.get("input_normalization", None),
        kernel_scale    = model_cfg.get("kernel_scale", 1.0),
        length_scale    = model_cfg.get("length_scale", 1.0),
    ).to(device)


@torch.no_grad()
def evaluate(model, loader, device, num_mc_samples=10):
    model.eval()
    running_loss = total_correct = total_examples = 0
    total_nll = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, variances = model(images, return_cov=True)
        probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        running_loss  += (-log_probs.gather(1, labels.unsqueeze(1)).mean()).item()
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll     += (-log_probs.gather(1, labels.unsqueeze(1)).sum()).item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t  = torch.cat(all_probs)
    all_labels_t = torch.cat(all_labels)
    return {
        "loss":     running_loss / len(loader),
        "accuracy": total_correct / total_examples,
        "nll":      total_nll / total_examples,
        "ece":      _classification_ece(all_probs_t, all_labels_t),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only pass for crashed april_4 experiments"
    )
    parser.add_argument(
        "--experiment", required=True,
        choices=list(_EXPERIMENT_REGISTRY.keys()),
        help="Which experiment type to eval (determines model class)",
    )
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--run-name",   required=True, dest="run_name",
                        help="W&B display name of the existing crashed run")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Seed number (used to reconstruct checkpoint path)")
    parser.add_argument("--checkpoint", default=None,
                        help="Direct path to best_model.pt (overrides auto-detection)")
    args = parser.parse_args()

    info = _EXPERIMENT_REGISTRY[args.experiment]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed

    # ── Find best_model.pt ──────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        base = Path(cfg["output"]["checkpoint_path"])
        if args.seed is not None:
            base = base.parent / f"seed{args.seed}" / base.name
        seed_dir = base.parent          # checkpoints_april4/<exp>/seed<N>
        ckpt_path = find_best_model(seed_dir)

    if ckpt_path is None or not ckpt_path.exists():
        print(f"ERROR: no checkpoint found. seed_dir={base.parent if args.checkpoint is None else '(direct)'}")
        sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Resume the existing W&B run ─────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    entity  = wandb_cfg.get("entity", "sta414manygp")
    project = wandb_cfg.get("project", "april_4_experiments")

    run_id = find_wandb_run_id(entity, project, args.run_name)
    if run_id is None:
        print(f"ERROR: could not find W&B run '{args.run_name}' in {entity}/{project}")
        sys.exit(1)

    import wandb
    run = wandb.init(project=project, entity=entity, id=run_id, resume="must")
    print(f"Resumed W&B run: {run.url}")

    # ── Data ────────────────────────────────────────────────────────────────────
    data_cfg   = cfg["data"]
    smoke_test = cfg["experiment"].get("smoke_test", False)
    _, _, val_loader, test_loader, _, val_dataset, test_dataset = get_cifar10_supcon_loaders(
        data_root   = data_cfg["root"],
        batch_size  = data_cfg["batch_size"],
        num_workers = data_cfg["num_workers"],
        smoke_test  = smoke_test,
    )
    print(f"Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # ── Model ───────────────────────────────────────────────────────────────────
    model = build_model(info["model_class"], cfg["model"], device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    stored_epoch = ckpt.get("epoch", "?")
    stored_acc   = ckpt.get("val_metrics", {}).get("accuracy", "?")
    print(f"Loaded weights from epoch {stored_epoch}  val_acc={stored_acc}")

    num_mc_samples = cfg.get("training", {}).get("num_mc_samples", 10)

    # ── Test set ────────────────────────────────────────────────────────────────
    print("\nEvaluating on held-out test set...")
    test_metrics = evaluate(model, test_loader, device, num_mc_samples)
    print(
        f"Test Acc: {test_metrics['accuracy']*100:.2f}%  "
        f"NLL: {test_metrics['nll']:.4f}  ECE: {test_metrics['ece']:.4f}"
    )
    run.log({
        "test/accuracy": test_metrics["accuracy"],
        "test/nll":      test_metrics["nll"],
        "test/ece":      test_metrics["ece"],
        "test/loss":     test_metrics["loss"],
    })

    # ── OOD + CIFAR-C ───────────────────────────────────────────────────────────
    if not smoke_test and cfg.get("ood", {}).get("enabled", True):
        print("\nRunning OOD + CIFAR-C evaluation...")
        from src.training.post_training_eval import run_full_ood_eval
        wrapper = ModelWrapper(
            model=model, has_cov=True,
            num_mc_samples=num_mc_samples, model_type="sngp_augmented",
        )
        id_logits, id_probs, _, _ = collect_logits_and_probs(
            wrapper, test_loader, device, num_mc_samples
        )
        run_full_ood_eval(
            model=model, has_cov=True,
            id_logits=id_logits, id_probs=id_probs,
            cfg=cfg, device=device, run=run,
            num_mc_samples=num_mc_samples, model_type="sngp_augmented",
        )

    # Mark the run so it's easy to spot in the W&B table
    run.summary["eval_only_pass"]           = True
    run.summary["checkpoint_epoch"]         = stored_epoch
    run.summary["checkpoint_val_accuracy"]  = stored_acc

    run.finish()
    print("\nDone — metrics logged to existing W&B run.")


if __name__ == "__main__":
    main()
