"""
Post-training OOD evaluation.

Runs OOD detection against SVHN and CIFAR-100 and logs metrics to W&B.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.cifar100 import get_cifar100_loaders
from src.data.svhn import get_svhn_loader
from src.training.ood_evaluate import collect_logits_and_probs, evaluate_ood_split
from src.utils.model_loader import ModelWrapper


def run_ood_eval(
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
    Run OOD detection on SVHN and CIFAR-100 and return nested metrics.
    """
    model.eval()
    wrapper = ModelWrapper(model=model, has_cov=has_cov, num_mc_samples=num_mc_samples, model_type=model_type)

    data_cfg = cfg["data"]
    data_root = data_cfg["root"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)
    ood_cfg = cfg.get("ood", {})
    id_normalization = ood_cfg.get("id_normalization", "cifar10")
    svhn_split = ood_cfg.get("svhn_split", "test")

    results: dict = {"ood": {}}

    print("\n=== OOD detection: SVHN ===")
    svhn_loader = get_svhn_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        id_normalization=id_normalization,
        split=svhn_split,
    )
    ood_logits, ood_probs, _, _ = collect_logits_and_probs(wrapper, svhn_loader, device, num_mc_samples)
    svhn_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs)
    results["ood"]["svhn"] = svhn_metrics
    print(
        f"  DS  AUROC: {svhn_metrics['dempster_shafer']['auroc']:.4f}  "
        f"AUPR: {svhn_metrics['dempster_shafer']['aupr']:.4f}"
    )
    print(
        f"  p_max AUROC: {svhn_metrics['max_prob']['auroc']:.4f}  "
        f"AUPR: {svhn_metrics['max_prob']['aupr']:.4f}"
    )

    if run is not None:
        run.log({
            "ood/svhn/ds_auroc": svhn_metrics["dempster_shafer"]["auroc"],
            "ood/svhn/ds_aupr": svhn_metrics["dempster_shafer"]["aupr"],
            "ood/svhn/mp_auroc": svhn_metrics["max_prob"]["auroc"],
            "ood/svhn/mp_aupr": svhn_metrics["max_prob"]["aupr"],
        })

    print("\n=== OOD detection: CIFAR-100 ===")
    _, cifar100_loader, _, _ = get_cifar100_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    ood_logits, ood_probs, _, _ = collect_logits_and_probs(wrapper, cifar100_loader, device, num_mc_samples)
    cifar100_metrics = evaluate_ood_split(id_logits, id_probs, ood_logits, ood_probs)
    results["ood"]["cifar100"] = cifar100_metrics
    print(
        f"  DS  AUROC: {cifar100_metrics['dempster_shafer']['auroc']:.4f}  "
        f"AUPR: {cifar100_metrics['dempster_shafer']['aupr']:.4f}"
    )
    print(
        f"  p_max AUROC: {cifar100_metrics['max_prob']['auroc']:.4f}  "
        f"AUPR: {cifar100_metrics['max_prob']['aupr']:.4f}"
    )

    if run is not None:
        run.log({
            "ood/cifar100/ds_auroc": cifar100_metrics["dempster_shafer"]["auroc"],
            "ood/cifar100/ds_aupr": cifar100_metrics["dempster_shafer"]["aupr"],
            "ood/cifar100/mp_auroc": cifar100_metrics["max_prob"]["auroc"],
            "ood/cifar100/mp_aupr": cifar100_metrics["max_prob"]["aupr"],
        })

    return results