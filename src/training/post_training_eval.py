"""
Post-training OOD and corruption robustness evaluation.

Runs the full evaluation suite in-process after training so that all metrics
land in the same W&B run:
  1. OOD detection vs SVHN (AUPR / AUROC, Dempster-Shafer + max-prob)
  2. OOD detection vs CIFAR-100 (AUPR / AUROC, Dempster-Shafer + max-prob)
  3. CIFAR-10-C corruption robustness (aggregate accuracy + ECE over all
     15 corruptions × 5 severities, plus per-corruption breakdown)

Call `run_full_ood_eval` at the end of a training script, passing the
already-loaded model so everything stays in a single W&B run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.cifar100 import get_cifar100_loaders
from src.data.cifarc import CORRUPTIONS, get_cifarc_loader, maybe_download_cifarc
from src.data.svhn import get_svhn_loader
from src.training.ood_evaluate import (
    collect_logits_and_probs,
    evaluate_cifarc_split,
    evaluate_ood_split,
)
from src.utils.model_loader import ModelWrapper


def run_full_ood_eval(
    model: nn.Module,
    has_cov: bool,
    id_logits: torch.Tensor,
    id_probs: torch.Tensor,
    cfg: dict,
    device: torch.device,
    run=None,
    num_mc_samples: int = 10,
    model_type: str = "",
) -> dict:
    """
    Run SVHN OOD, CIFAR-100 OOD, and CIFAR-10-C evaluations and log to W&B.

    The caller is responsible for collecting `id_logits` and `id_probs` from
    the CIFAR-10 test set before calling this function (so the test-set
    inference is not repeated).

    Args:
        model:           Trained model already loaded in eval mode.
        has_cov:         Whether the model exposes GP covariance (return_cov).
        id_logits:       (N, K) logits collected on the CIFAR-10 test set.
        id_probs:        (N, K) predictive probs for the CIFAR-10 test set.
        cfg:             Full experiment config dict (needs data.root, ood.*).
        device:          Torch device.
        run:             W&B run object (or None if W&B is disabled).
        num_mc_samples:  MC samples for Laplace approximation.
        model_type:      Descriptive string for the wrapper (e.g. "sngp").

    Returns:
        Nested dict with all OOD and CIFAR-C metrics.
    """
    model.eval()
    wrapper = ModelWrapper(model=model, has_cov=has_cov, num_mc_samples=num_mc_samples, model_type=model_type)

    data_cfg = cfg["data"]
    data_root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)
    ood_cfg = cfg.get("ood", {})

    results: dict = {"ood": {}, "cifarc": {}}
    is_due = model_type == "due"
    # -------------------------------------------------------------------------
    # 1. OOD detection: SVHN
    # -------------------------------------------------------------------------
    print("\n=== OOD detection: SVHN ===")
    svhn_loader = get_svhn_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        id_normalization="cifar10",
    )
    ood_logits, ood_probs, _, _ = collect_logits_and_probs(wrapper, svhn_loader, device, num_mc_samples)
    svhn_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs, is_due)
    results["ood"]["svhn"] = svhn_metrics
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

    # -------------------------------------------------------------------------
    # 2. OOD detection: CIFAR-100
    # -------------------------------------------------------------------------
    print("\n=== OOD detection: CIFAR-100 ===")
    _, cifar100_loader, _, _ = get_cifar100_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    ood_logits, ood_probs, _, _ = collect_logits_and_probs(wrapper, cifar100_loader, device, num_mc_samples)
    cifar100_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs, is_due)
    results["ood"]["cifar100"] = cifar100_metrics
    print(f"  DS  AUROC: {cifar100_metrics['dempster_shafer']['auroc']:.4f}  "
          f"AUPR: {cifar100_metrics['dempster_shafer']['aupr']:.4f}")
    print(f"  p_max AUROC: {cifar100_metrics['max_prob']['auroc']:.4f}  "
          f"AUPR: {cifar100_metrics['max_prob']['aupr']:.4f}")

    if run is not None:
        run.log({
            "ood/cifar100/ds_auroc":  cifar100_metrics["dempster_shafer"]["auroc"],
            "ood/cifar100/ds_aupr":   cifar100_metrics["dempster_shafer"]["aupr"],
            "ood/cifar100/mp_auroc":  cifar100_metrics["max_prob"]["auroc"],
            "ood/cifar100/mp_aupr":   cifar100_metrics["max_prob"]["aupr"],
        })

    # -------------------------------------------------------------------------
    # 3. CIFAR-10-C corruption robustness
    # -------------------------------------------------------------------------
    print("\n=== CIFAR-10-C corruption robustness ===")
    cifarc_root = ood_cfg.get("cifarc_root", data_root)
    cifarc_dir = maybe_download_cifarc(cifarc_root, "cifar10")
    corruptions_to_eval = ood_cfg.get("corruptions") or CORRUPTIONS
    severities_to_eval = ood_cfg.get("severities") or [1, 2, 3, 4, 5]

    all_accuracies: list[float] = []
    all_eces: list[float] = []

    from tqdm.auto import tqdm
    for corruption in tqdm(corruptions_to_eval, desc="CIFAR-C Corruptions"):
        results["cifarc"][corruption] = {}
        for severity in severities_to_eval:
            cifarc_loader, cifarc_labels = get_cifarc_loader(
                cifarc_dir=cifarc_dir,
                corruption=corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=num_workers,
                id_normalization="cifar10",
            )
            _, probs, _, _ = collect_logits_and_probs(wrapper, cifarc_loader, device, num_mc_samples)
            split_metrics = evaluate_cifarc_split(probs, cifarc_labels)
            results["cifarc"][corruption][str(severity)] = split_metrics

            all_accuracies.append(split_metrics["accuracy"])
            all_eces.append(split_metrics["ece"])

            if run is not None:
                run.log({
                    f"cifarc/{corruption}/severity_{severity}/accuracy": split_metrics["accuracy"],
                    f"cifarc/{corruption}/severity_{severity}/ece":      split_metrics["ece"],
                    f"cifarc/{corruption}/severity_{severity}/mce":      split_metrics["mce"],
                })

    # Aggregate over all corruptions × severities (matches "Corrupted" column in paper)
    mean_corrupted_acc = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    mean_corrupted_ece = sum(all_eces) / len(all_eces) if all_eces else 0.0
    results["cifarc"]["mean_accuracy"] = mean_corrupted_acc
    results["cifarc"]["mean_ece"] = mean_corrupted_ece

    print(f"\n  Mean corrupted accuracy: {mean_corrupted_acc * 100:.2f}%")
    print(f"  Mean corrupted ECE:      {mean_corrupted_ece:.4f}")

    if run is not None:
        run.log({
            "test/corrupted_accuracy": mean_corrupted_acc,
            "test/corrupted_ece":      mean_corrupted_ece,
        })

    return results
