"""
Deep Ensemble evaluation for CIFAR-10.

Loads M independently trained CifarResNetClassifier checkpoints (one per seed),
averages their softmax predictions at test time, and evaluates:
  - Clean test accuracy, NLL, ECE
  - CIFAR-10-C corrupted accuracy and ECE
  - OOD AUPR/AUROC vs SVHN and CIFAR-100

All results are logged to W&B under the run name "deep_ensemble".

Usage:
    python experiments/cifar10_deep_ensemble_eval.py \
        --checkpoints ./checkpoints_april4/deterministic/seed*/best_model.pt \
        --config configs/experiment_april4_deterministic.yaml

The --checkpoints argument accepts shell glob patterns (quoted) or space-separated paths.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.data.cifar100 import get_cifar100_loaders
from src.data.svhn import get_svhn_loader
from src.data.cifarc import CORRUPTIONS, get_cifarc_loader, maybe_download_cifarc
from src.models.resnet import CifarResNetClassifier
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import evaluate_ood_split, evaluate_cifarc_split


# ---------------------------------------------------------------------------
# Ensemble inference helpers
# ---------------------------------------------------------------------------

def load_member(checkpoint_path: str, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CifarResNetClassifier(
        widen_factor=model_cfg.get("widen_factor", 10),
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def ensemble_probs(models: list[torch.nn.Module], loader, device: torch.device):
    """Return (N, K) averaged softmax probs and (N,) labels over the full loader."""
    all_avg_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        images, labels = batch[0].to(device, non_blocking=True), batch[1]
        member_probs = torch.stack(
            [F.softmax(m(images), dim=-1).cpu() for m in models], dim=0
        )  # (M, B, K)
        avg = member_probs.mean(dim=0)  # (B, K)
        all_avg_probs.append(avg)
        all_labels.append(labels)

    return torch.cat(all_avg_probs), torch.cat(all_labels)


@torch.no_grad()
def ensemble_probs_unlabeled(models: list[torch.nn.Module], loader, device: torch.device):
    """Return (N, K) averaged softmax probs for a loader with no labels (OOD data)."""
    all_avg_probs: list[torch.Tensor] = []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device, non_blocking=True)
        else:
            images = batch.to(device, non_blocking=True)
        member_probs = torch.stack(
            [F.softmax(m(images), dim=-1).cpu() for m in models], dim=0
        )
        all_avg_probs.append(member_probs.mean(dim=0))

    return torch.cat(all_avg_probs)


def compute_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    log_probs = probs.clamp_min(1e-12).log()
    return -log_probs.gather(1, labels.unsqueeze(1)).mean().item()


# ---------------------------------------------------------------------------
# Pseudo-logits for DS uncertainty: log(avg_probs) scaled back
# (DS uncertainty is K / (K + sum exp(logits)); using log_avg_probs as logits
#  gives a meaningful entropy-like signal consistent with the other experiments)
# ---------------------------------------------------------------------------

def probs_to_pseudo_logits(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(1e-12).log()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Ensemble evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to best_model.pt checkpoints (one per seed). "
                             "Accepts glob patterns.")
    parser.add_argument("--config", required=True, help="Path to the deterministic YAML config")
    parser.add_argument("--run-name", default="deep_ensemble", dest="run_name")
    args = parser.parse_args()

    # Expand any glob patterns
    checkpoint_paths: list[str] = []
    for pat in args.checkpoints:
        expanded = glob_module.glob(pat)
        checkpoint_paths.extend(sorted(expanded) if expanded else [pat])

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found matching: {args.checkpoints}")

    print(f"Deep Ensemble: {len(checkpoint_paths)} members")
    for p in checkpoint_paths:
        print(f"  {p}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    models = [load_member(p, model_cfg, device) for p in checkpoint_paths]
    print(f"Loaded {len(models)} ensemble members.")

    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        run = wandb.init(
            project=wandb_cfg.get("project", "april_4_experiments"),
            entity=wandb_cfg.get("entity") or "sta414manygp",
            name=args.run_name,
            config={"ensemble_size": len(models), "checkpoints": checkpoint_paths, **cfg},
        )

    data_cfg = cfg["data"]
    smoke_test = cfg["experiment"].get("smoke_test", False)
    _, _, test_loader, _, _, test_dataset = get_cifar10_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )

    # ── Clean test evaluation ─────────────────────────────────────────────────
    print("\nEvaluating ensemble on CIFAR-10 test set...")
    test_probs, test_labels = ensemble_probs(models, test_loader, device)
    test_acc = (test_probs.argmax(dim=1) == test_labels).float().mean().item()
    test_nll = compute_nll(test_probs, test_labels)
    test_ece = _classification_ece(test_probs, test_labels)
    print(f"Test Acc: {test_acc * 100:.2f}%  NLL: {test_nll:.4f}  ECE: {test_ece:.4f}")

    if run is not None:
        run.log({"test/accuracy": test_acc, "test/nll": test_nll, "test/ece": test_ece})

    id_logits = probs_to_pseudo_logits(test_probs)
    id_probs  = test_probs

    ood_cfg  = cfg.get("ood", {})
    batch_size   = data_cfg["batch_size"]
    num_workers  = data_cfg.get("num_workers", 2)
    data_root    = data_cfg["root"]

    if not smoke_test and ood_cfg.get("enabled", True):
        # ── OOD: SVHN ────────────────────────────────────────────────────────
        print("\n=== OOD detection: SVHN ===")
        svhn_loader = get_svhn_loader(
            data_root=data_root, batch_size=batch_size,
            num_workers=num_workers, id_normalization="cifar10",
        )
        ood_probs_svhn = ensemble_probs_unlabeled(models, svhn_loader, device)
        ood_logits_svhn = probs_to_pseudo_logits(ood_probs_svhn)
        svhn_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits_svhn, ood_probs_svhn)
        print(f"  DS  AUROC: {svhn_metrics['dempster_shafer']['auroc']:.4f}  "
              f"AUPR: {svhn_metrics['dempster_shafer']['aupr']:.4f}")
        print(f"  p_max AUROC: {svhn_metrics['max_prob']['auroc']:.4f}  "
              f"AUPR: {svhn_metrics['max_prob']['aupr']:.4f}")

        if run is not None:
            run.log({
                "ood/svhn/ds_auroc":  svhn_metrics["dempster_shafer"]["auroc"],
                "ood/svhn/ds_aupr":   svhn_metrics["dempster_shafer"]["aupr"],
                "ood/svhn/mp_auroc":  svhn_metrics["max_prob"]["auroc"],
                "ood/svhn/mp_aupr":   svhn_metrics["max_prob"]["aupr"],
            })

        # ── OOD: CIFAR-100 ───────────────────────────────────────────────────
        print("\n=== OOD detection: CIFAR-100 ===")
        _, cifar100_loader, _, _ = get_cifar100_loaders(
            data_root=data_root, batch_size=batch_size, num_workers=num_workers,
        )
        ood_probs_c100 = ensemble_probs_unlabeled(models, cifar100_loader, device)
        ood_logits_c100 = probs_to_pseudo_logits(ood_probs_c100)
        c100_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits_c100, ood_probs_c100)
        print(f"  DS  AUROC: {c100_metrics['dempster_shafer']['auroc']:.4f}  "
              f"AUPR: {c100_metrics['dempster_shafer']['aupr']:.4f}")
        print(f"  p_max AUROC: {c100_metrics['max_prob']['auroc']:.4f}  "
              f"AUPR: {c100_metrics['max_prob']['aupr']:.4f}")

        if run is not None:
            run.log({
                "ood/cifar100/ds_auroc":  c100_metrics["dempster_shafer"]["auroc"],
                "ood/cifar100/ds_aupr":   c100_metrics["dempster_shafer"]["aupr"],
                "ood/cifar100/mp_auroc":  c100_metrics["max_prob"]["auroc"],
                "ood/cifar100/mp_aupr":   c100_metrics["max_prob"]["aupr"],
            })

        # ── CIFAR-10-C ───────────────────────────────────────────────────────
        print("\n=== CIFAR-10-C corruption robustness ===")
        cifarc_root = ood_cfg.get("cifarc_root", data_root)
        cifarc_dir  = maybe_download_cifarc(cifarc_root, "cifar10")
        corruptions = ood_cfg.get("corruptions") or CORRUPTIONS
        severities  = ood_cfg.get("severities") or [1, 2, 3, 4, 5]

        all_acc, all_ece = [], []
        from tqdm.auto import tqdm
        for corruption in tqdm(corruptions, desc="CIFAR-C"):
            for severity in severities:
                cifarc_loader, cifarc_labels = get_cifarc_loader(
                    cifarc_dir=cifarc_dir, corruption=corruption, severity=severity,
                    batch_size=batch_size, num_workers=num_workers, id_normalization="cifar10",
                )
                c_probs = ensemble_probs_unlabeled(models, cifarc_loader, device)
                split_metrics = evaluate_cifarc_split(c_probs, cifarc_labels)
                all_acc.append(split_metrics["accuracy"])
                all_ece.append(split_metrics["ece"])

                if run is not None:
                    run.log({
                        f"cifarc/{corruption}/severity_{severity}/accuracy": split_metrics["accuracy"],
                        f"cifarc/{corruption}/severity_{severity}/ece":      split_metrics["ece"],
                    })

        mean_c_acc = sum(all_acc) / len(all_acc)
        mean_c_ece = sum(all_ece) / len(all_ece)
        print(f"\n  Mean corrupted accuracy: {mean_c_acc * 100:.2f}%")
        print(f"  Mean corrupted ECE:      {mean_c_ece:.4f}")

        if run is not None:
            run.log({
                "test/corrupted_accuracy": mean_c_acc,
                "test/corrupted_ece":      mean_c_ece,
            })

    if run is not None:
        run.finish()

    print("\n=== Deep Ensemble Summary ===")
    print(f"  Members:            {len(models)}")
    print(f"  Test accuracy:      {test_acc * 100:.2f}%")
    print(f"  Test NLL:           {test_nll:.4f}")
    print(f"  Test ECE:           {test_ece:.4f}")


if __name__ == "__main__":
    main()
