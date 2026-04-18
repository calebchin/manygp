"""
Eval-only pass for a crashed/incomplete sngp_augmented (or rff variant) run.

Loads the best_model.pt checkpoint saved during training, then runs the full
post-training evaluation suite (test set, OOD detection, CIFAR-10-C) and logs
all metrics back into the EXISTING W&B run — identified by run display name —
so the results appear on the same run that crashed rather than as a new entry.

Typical use case: the training job finished successfully (best_model.pt exists)
but the job was killed by the scheduler before the OOD/CIFAR-C block could run.

Usage:
    python experiments/eval_only_sngp_aug.py \\
        --config  configs/experiment_april2_sngp_augmented.yaml \\
        --seed    0 \\
        --num-inducing 4096 \\
        --run-name sngp_aug_rff4096_seed0

    # Or point directly at a checkpoint:
    python experiments/eval_only_sngp_aug.py \\
        --config    configs/experiment_april2_sngp_augmented.yaml \\
        --checkpoint /w/20252/davida/manygp/manygp/checkpoints_april2/sngp_aug_rff4096/seed0/best_model.pt \\
        --run-name  sngp_aug_rff4096_seed0
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
from src.models.sngp import SNGPResNetClassifier, laplace_predictive_probs
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper


# ── W&B run lookup ─────────────────────────────────────────────────────────────

def find_wandb_run_id(entity: str, project: str, run_name: str) -> str | None:
    """Return the W&B internal run ID for a run with the given display name."""
    import wandb
    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": run_name},
        per_page=10,
    )
    for run in runs:
        if run.name == run_name:
            print(f"  Found W&B run '{run_name}' → id={run.id}  state={run.state}")
            return run.id
    return None


# ── Eval ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sngp(model, loader, device, num_mc_samples: int = 10) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_nll = 0.0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, variances = model(images, return_cov=True)
        probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        loss = -log_probs.gather(1, labels.unsqueeze(1)).mean()

        running_loss += loss.item()
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll += -log_probs.gather(1, labels.unsqueeze(1)).sum().item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t = torch.cat(all_probs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    ece = _classification_ece(all_probs_t, all_labels_t)
    return {
        "loss":     running_loss / len(loader),
        "accuracy": total_correct / total_examples,
        "nll":      total_nll / total_examples,
        "ece":      ece,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only pass for a crashed sngp_augmented / rff run"
    )
    parser.add_argument("--config",        required=True, help="Path to YAML config")
    parser.add_argument("--run-name",      required=True, dest="run_name",
                        help="W&B display name of the existing crashed run "
                             "(e.g. sngp_aug_rff4096_seed0)")
    parser.add_argument("--seed",          type=int, default=None,
                        help="Seed used in training — used to reconstruct checkpoint path")
    parser.add_argument("--num-inducing",  type=int, default=None, dest="num_inducing",
                        help="RFF num_inducing used in training — used to reconstruct checkpoint path")
    parser.add_argument("--checkpoint",    default=None,
                        help="Direct path to best_model.pt (overrides auto-reconstruction)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Reconstruct checkpoint path (mirrors cifar10_sngp_augmented.py logic) ──
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        base = Path(cfg["output"]["checkpoint_path"])
        if args.seed is not None:
            base = base.parent / f"seed{args.seed}" / base.name
        if args.num_inducing is not None:
            # mirrors training script logic:
            # base.parent        = checkpoints_april2/sngp_augmented/seed0
            # base.parent.parent = checkpoints_april2/sngp_augmented
            # rff_dir            = checkpoints_april2/sngp_augmented/sngp_aug_rff4096
            rff_dir = base.parent.parent / f"sngp_aug_rff{args.num_inducing}"
            seed_part = base.parent.name  # e.g. "seed0"
            base = rff_dir / seed_part / base.name
        ckpt_path = base.parent / "best_model.pt"

    if not ckpt_path.exists():
        # Try all top-k candidates in the same directory
        candidates = sorted(ckpt_path.parent.glob("*.pt"))
        # Exclude best_model.pt itself (already checked) and pick the one with
        # the best val_accuracy baked into its filename.
        candidates = [c for c in candidates if c.name != "best_model.pt"]
        if not candidates:
            print(f"ERROR: no checkpoint found at {ckpt_path} or in {ckpt_path.parent}")
            sys.exit(1)
        # Pick the checkpoint with the highest accuracy value in its name
        def _acc_from_name(p: Path) -> float:
            import re
            m = re.search(r"val_accuracy([\d.]+)", p.name)
            return float(m.group(1)) if m else 0.0
        ckpt_path = max(candidates, key=_acc_from_name)
        print(f"best_model.pt not found — using: {ckpt_path}")
    else:
        print(f"Checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Apply config overrides ──────────────────────────────────────────────────
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
    if args.num_inducing is not None:
        cfg.setdefault("model", {})["num_inducing"] = args.num_inducing
        cfg.setdefault("experiment", {})["name"] = f"cifar10_sngp_aug_rff{args.num_inducing}"

    # ── Resume the existing W&B run ─────────────────────────────────────────────
    wandb_cfg = cfg.get("wandb", {})
    entity  = wandb_cfg.get("entity", "sta414manygp")
    project = wandb_cfg.get("project", "april_4_experiments")

    run_id = find_wandb_run_id(entity, project, args.run_name)
    if run_id is None:
        print(f"ERROR: could not find W&B run with display name '{args.run_name}' "
              f"in {entity}/{project}")
        sys.exit(1)

    import wandb
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must",   # fail loudly if the run doesn't exist
    )
    print(f"Resumed W&B run: {run.url}")

    # ── Data ────────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    smoke_test = cfg["experiment"].get("smoke_test", False)
    _, _, val_loader, test_loader, _, val_dataset, test_dataset = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    # ── Model ───────────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dim=model_cfg["hidden_dim"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}  "
          f"val_accuracy={ckpt.get('val_accuracy', '?')}")

    num_mc_samples = cfg.get("training", {}).get("num_mc_samples", 10)

    # ── Test set ────────────────────────────────────────────────────────────────
    print("\nEvaluating on held-out test set...")
    test_metrics = evaluate_sngp(model, test_loader, device, num_mc_samples)
    print(
        f"Test Acc: {test_metrics['accuracy'] * 100:.2f}%  "
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
            model=model, has_cov=True, num_mc_samples=num_mc_samples,
            model_type="sngp_augmented",
        )
        id_logits, id_probs, _, _ = collect_logits_and_probs(
            wrapper, test_loader, device, num_mc_samples
        )
        run_full_ood_eval(
            model=model, has_cov=True, id_logits=id_logits, id_probs=id_probs,
            cfg=cfg, device=device, run=run, num_mc_samples=num_mc_samples,
            model_type="sngp_augmented",
        )

    # Log checkpoint epoch as a summary so it's visible in the run
    run.summary["eval_only_pass"] = True
    run.summary["checkpoint_epoch"] = ckpt.get("epoch", -1)
    run.summary["checkpoint_val_accuracy"] = ckpt.get("val_accuracy", float("nan"))

    run.finish()
    print("\nDone — results logged to existing W&B run.")


if __name__ == "__main__":
    main()
