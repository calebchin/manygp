"""
OOD detection evaluation metrics and batched inference helpers.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader


def dempster_shafer_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """
    Dempster-Shafer uncertainty score.

    Higher values indicate more uncertainty (more likely OOD).
    """
    num_classes = logits.shape[-1]
    evidence = logits.exp().sum(dim=-1)
    return num_classes / (num_classes + evidence)


def max_prob_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """
    Max-probability uncertainty score: 1 - max(prob).

    Higher values indicate more uncertainty (more likely OOD).
    """
    return 1.0 - probs.max(dim=-1).values


def compute_ood_metrics(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict[str, float]:
    """
    Computes AUROC and AUPR for binary OOD detection.

    ID samples are labeled 0 and OOD samples are labeled 1.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = torch.cat([
        torch.zeros(len(id_scores), dtype=torch.float32),
        torch.ones(len(ood_scores), dtype=torch.float32),
    ]).numpy()
    scores = torch.cat([id_scores.cpu().float(), ood_scores.cpu().float()]).numpy()

    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    return {"auroc": auroc, "aupr": aupr}


@torch.no_grad()
def collect_logits_and_probs(
    model_wrapper,
    loader: DataLoader,
    device: torch.device,
    num_mc_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Runs full inference over a loader and collects logits/probs/labels.

    Returns:
        logits, probs, labels, latency_ms_per_image
    """
    all_logits = []
    all_probs = []
    all_labels = []
    total_images = 0

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        logits, variances = model_wrapper.predict_logits(images)

        if model_wrapper.has_cov and variances is not None:
            from src.models.sngp import laplace_predictive_probs

            probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        else:
            probs = torch.softmax(logits, dim=-1)

        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        total_images += labels.size(0)

    if use_cuda:
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    latency_ms_per_image = elapsed_ms / max(total_images, 1)

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_probs, dim=0),
        torch.cat(all_labels, dim=0),
        latency_ms_per_image,
    )


def evaluate_ood_split(
    id_logits: torch.Tensor,
    id_probs: torch.Tensor,
    ood_logits: torch.Tensor,
    ood_probs: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """
    Evaluates OOD detection with Dempster-Shafer and max-prob baselines.
    """
    ds_id = dempster_shafer_uncertainty(id_logits)
    ds_ood = dempster_shafer_uncertainty(ood_logits)
    mp_id = max_prob_uncertainty(id_probs)
    mp_ood = max_prob_uncertainty(ood_probs)

    return {
        "dempster_shafer": compute_ood_metrics(ds_id, ds_ood),
        "max_prob": compute_ood_metrics(mp_id, mp_ood),
    }