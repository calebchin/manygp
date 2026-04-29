"""
OOD detection and corruption robustness evaluation metrics.

Provides:
  - dempster_shafer_uncertainty: uncertainty score from logits (DS theory)
  - max_prob_uncertainty:        1 - max(softmax) baseline
  - compute_ood_metrics:         AUROC + AUPR given ID/OOD uncertainty scores
  - collect_logits_and_probs:    batched inference with GPU-synced latency
  - evaluate_ood_split:          runs both scoring methods for one OOD dataset
  - evaluate_cifarc_split:       accuracy + ECE + MCE for one corruption/severity
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def dempster_shafer_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """
    Dempster-Shafer uncertainty score.

    u(x) = K / (K + sum_k exp(h_k(x)))

    Higher values indicate more uncertainty (more likely OOD).

    Args:
        logits: (N, K) raw (pre-softmax) logits.

    Returns:
        (N,) uncertainty scores in (0, 1).
    """
    K = logits.shape[-1]
    evidence = logits.exp().sum(dim=-1)
    return K / (K + evidence)


def max_prob_uncertainty(probs: torch.Tensor) -> torch.Tensor:
    """
    Max-probability uncertainty: 1 - max(softmax).

    Higher values indicate more uncertainty (more likely OOD).

    Args:
        probs: (N, K) softmax probabilities.

    Returns:
        (N,) uncertainty scores in (0, 1).
    """
    return 1.0 - probs.max(dim=-1).values


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> dict[str, float]:
    """
    Computes AUROC and AUPR for binary OOD detection.

    ID samples are labeled 0, OOD samples are labeled 1.
    Higher uncertainty score = predicted OOD.

    Args:
        id_scores:  (N_id,) uncertainty scores for in-distribution samples.
        ood_scores: (N_ood,) uncertainty scores for OOD samples.

    Returns:
        {'auroc': float, 'aupr': float}
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
    Runs full inference over the loader and collects logits, probabilities,
    labels, and inference latency.

    GPU timing uses torch.cuda.synchronize() to measure actual kernel time,
    not just launch time.

    Args:
        model_wrapper:   ModelWrapper from src.utils.model_loader.
        loader:          DataLoader yielding (images, labels).
        device:          Target device.
        num_mc_samples:  MC samples for SNGP Laplace approximation.

    Returns:
        (logits, probs, labels, latency_ms_per_image)
        logits:  (N, K) — raw logits (first output, before Laplace sampling)
        probs:   (N, K) — predictive probabilities
        labels:  (N,)   — ground-truth class indices
        latency_ms_per_image: float
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

        if model_wrapper.model_type == "due":
            logits, probs = model_wrapper.predict_probs(images)
        else:
            logits, variances = model_wrapper.predict_logits(images)

            if model_wrapper.has_cov and variances is not None:
                from src.models.sngp import laplace_predictive_probs
                probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
            else:
                probs = F.softmax(logits, dim=-1)

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
    due: Bool = False,
) -> dict[str, dict[str, float]]:
    """
    Evaluates OOD detection for one OOD dataset using both scoring methods.

    Args:
        id_logits:  (N_id, K)  logits for in-distribution test set.
        id_probs:   (N_id, K)  predictive probs for in-distribution test set.
        ood_logits: (N_ood, K) logits for OOD dataset.
        ood_probs:  (N_ood, K) predictive probs for OOD dataset.

    Returns:
        {
          "dempster_shafer": {"auroc": float, "aupr": float},
          "max_prob":        {"auroc": float, "aupr": float},
        }
    """
    if due:
      ds_id = id_logits
      ds_ood = ood_logits
    else:
        ds_id = dempster_shafer_uncertainty(id_logits)
        ds_ood = dempster_shafer_uncertainty(ood_logits)
    mp_id = max_prob_uncertainty(id_probs)
    mp_ood = max_prob_uncertainty(ood_probs)

    return {
        "dempster_shafer": compute_ood_metrics(ds_id, ds_ood),
        "max_prob": compute_ood_metrics(mp_id, mp_ood),
    }


def evaluate_cifarc_split(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """
    Computes accuracy, ECE, and MCE for one corruption type and severity.

    Args:
        probs:  (N, K) predictive probabilities.
        labels: (N,)   ground-truth class indices.

    Returns:
        {"accuracy": float, "ece": float, "mce": float}
    """
    from src.training.evaluate import _classification_ece, _classification_mce

    preds = probs.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()
    ece = _classification_ece(probs, labels)
    mce = _classification_mce(probs, labels)
    return {"accuracy": accuracy, "ece": ece, "mce": mce}
