import os
import tempfile

import torch
from torch.utils.data import DataLoader


def _classification_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for multi-class classification (Guo et al. 2017).

    Args:
        probs:  (N, C) predicted class probabilities
        labels: (N,)   ground-truth class indices
        n_bins: number of equal-width confidence bins

    Returns:
        ECE as a float in [0, 1]
    """
    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = confidences[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(bin_acc - bin_conf)
    return ece


def _classification_mce(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Maximum Calibration Error for multi-class classification.

    Same binning as ECE but returns the worst-case bin gap instead of the weighted average.

    Args:
        probs:  (N, C) predicted class probabilities
        labels: (N,)   ground-truth class indices
        n_bins: number of equal-width confidence bins

    Returns:
        MCE as a float in [0, 1]
    """
    confidences, preds = probs.max(dim=-1)
    correct = (preds == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = confidences[mask].mean().item()
        mce = max(mce, abs(bin_acc - bin_conf))
    return mce


def _regression_mce(mus: torch.Tensor, variances: torch.Tensor, targets: torch.Tensor, n_bins: int = 19) -> float:
    """
    Maximum Calibration Error for regression via predictive interval coverage.

    Same as regression ECE but returns the worst-case deviation across all quantile levels.

    Args:
        mus:       (N,) predictive means
        variances: (N,) predictive variances
        targets:   (N,) ground-truth values
        n_bins:    number of confidence levels to evaluate (default 19 → 0.05..0.95 step 0.05)

    Returns:
        MCE as a float in [0, 1]
    """
    from scipy.stats import norm as scipy_norm

    alphas = torch.linspace(0.05, 0.95, n_bins)
    stds = variances.sqrt()
    mce = 0.0
    for alpha in alphas:
        z = float(scipy_norm.ppf(0.5 + alpha.item() / 2))
        lo = mus - z * stds
        hi = mus + z * stds
        coverage = ((targets >= lo) & (targets <= hi)).float().mean().item()
        mce = max(mce, abs(coverage - alpha.item()))
    return mce


def _regression_ece(mus: torch.Tensor, variances: torch.Tensor, targets: torch.Tensor, n_bins: int = 19) -> float:
    """
    Expected Calibration Error for regression via predictive interval coverage (Kuleshov et al. 2018).

    For each confidence level alpha in (0, 1), computes the fraction of targets
    that fall within the symmetric (1-alpha) credible interval of the Gaussian
    predictive distribution, then averages |empirical_coverage - alpha| over all levels.

    Args:
        mus:       (N,) predictive means
        variances: (N,) predictive variances
        targets:   (N,) ground-truth values
        n_bins:    number of confidence levels to evaluate (default 19 → 0.05..0.95 step 0.05)

    Returns:
        ECE as a float in [0, 1]
    """
    from scipy.stats import norm as scipy_norm

    alphas = torch.linspace(0.05, 0.95, n_bins)
    stds = variances.sqrt()
    ece = 0.0
    for alpha in alphas:
        z = float(scipy_norm.ppf(0.5 + alpha.item() / 2))
        lo = mus - z * stds
        hi = mus + z * stds
        coverage = ((targets >= lo) & (targets <= hi)).float().mean().item()
        ece += abs(coverage - alpha.item())
    return ece / len(alphas)


def evaluate_classifier(model, test_loader: DataLoader, test_y: torch.Tensor, feature_extractor, dataset_name, device, run=None) -> dict:
    """
    Evaluate a TwoLayerDSPPClassifier on a test set.

    Args:
        model:             PyTorch model (classifier)
        test_loader:       DataLoader yielding (x_batch, y_batch)
        test_y:            Ground-truth label tensor (N,) on any device
        feature_extractor: Feature extractor (PyTorch model)
        dataset_name:      Name of dataset being evaluated (for wandb logs)
        device:            torch.device
        run:               Optional W&B run object. If provided, logs eval metrics and
                           saves a predictions artifact.

    Returns:
        {'accuracy': float, 'nll': float, 'ece': float}
        accuracy is a fraction in [0, 1]; nll is in nats; ece is in [0, 1].
    """
    preds, log_probs, probs = model.predict(test_loader, cnn=feature_extractor, device=device)

    # Collect ground-truth labels
    all_labels = []
    for _, y_batch in test_loader:
        all_labels.append(y_batch)
    labels = torch.cat(all_labels)
    print(preds[:10], labels[:10])

    accuracy = (preds == labels).float().mean().item()
    nll = -log_probs.mean().item()
    ece = _classification_ece(probs, labels)
    mce = _classification_mce(probs, labels)

    if run is not None:
        import wandb
        run.log({
            f"{dataset_name}/eval/accuracy": accuracy,
            f"{dataset_name}/eval/nll": nll,
            f"{dataset_name}/eval/ece": ece,
            f"{dataset_name}/eval/mce": mce,
        })
        preds_artifact = wandb.Artifact("predictions", type="evaluation")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"preds": preds, "log_probs": log_probs, "probs": probs, "labels": labels}, f.name)
            tmp_path = f.name
        preds_artifact.add_file(tmp_path, name=f"{dataset_name}_test_predictions.pt")
        run.log_artifact(preds_artifact)
        os.unlink(tmp_path)

    return {"accuracy": accuracy, "nll": nll, "ece": ece, "mce": mce}


def evaluate_regressor(model, test_loader: DataLoader, test_y: torch.Tensor, dataset_name, run=None) -> dict:
    """
    Evaluate a TwoLayerDSPP regression model on a test set.

    Fixes the missing evaluation cell from the original dspp.ipynb.

    Args:
        model:        TwoLayerDSPP (must be in eval mode)
        test_loader:  DataLoader yielding (x_batch, y_batch)
        test_y:       Ground-truth target tensor (N,) on any device
        dataset_name: Name of dataset being evaluated (for wandb logs)
        run:          Optional W&B run object. If provided, logs eval metrics and
                      saves a predictions artifact.

    Returns:
        {'rmse': float, 'nll': float, 'ece': float}
        rmse is on the normalised label scale; nll is in nats; ece is in [0, 1].
    """
    model.eval()
    mus, variances, lls = model.predict(test_loader)

    test_y_cpu = test_y.cpu()
    rmse = (mus - test_y_cpu).pow(2).mean().sqrt().item()
    nll = -lls.mean().item()
    ece = _regression_ece(mus, variances, test_y_cpu)
    mce = _regression_mce(mus, variances, test_y_cpu)

    if run is not None:
        import wandb
        run.log({
            f"{dataset_name}/eval/rmse": rmse,
            f"{dataset_name}/eval/nll": nll,
            f"{dataset_name}/eval/ece": ece,
            f"{dataset_name}/eval/mce": mce,
        })
        preds_artifact = wandb.Artifact("predictions", type="evaluation")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"mus": mus, "variances": variances, "lls": lls}, f.name)
            tmp_path = f.name
        preds_artifact.add_file(tmp_path, name=f"{dataset_name}_test_predictions.pt")
        run.log_artifact(preds_artifact)
        os.unlink(tmp_path)

    return {"rmse": rmse, "nll": nll, "ece": ece, "mce": mce}
