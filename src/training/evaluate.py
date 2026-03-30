import os
import tempfile

import torch
from torch.utils.data import DataLoader


def evaluate_classifier(model, test_loader: DataLoader, cnn, device, run=None) -> dict:
    """
    Evaluate a TwoLayerDSPPClassifier on a test set.

    Args:
        model:       TwoLayerDSPPClassifier
        test_loader: DataLoader yielding (x_batch, y_batch)
        cnn:         Feature extractor (CNNFeatureExtractor)
        device:      torch.device
        run:         Optional W&B run object. If provided, logs eval metrics and
                     saves a predictions artifact.

    Returns:
        {'accuracy': float, 'nll': float}
        accuracy is a fraction in [0, 1]; nll is in nats.
    """
    preds, log_probs = model.predict(test_loader, cnn=cnn, device=device)

    # Collect ground-truth labels
    all_labels = []
    for _, y_batch in test_loader:
        all_labels.append(y_batch)
    labels = torch.cat(all_labels)

    accuracy = (preds == labels).float().mean().item()
    nll = -log_probs.mean().item()

    if run is not None:
        import wandb
        run.log({"eval/accuracy": accuracy, "eval/nll": nll})
        preds_artifact = wandb.Artifact("predictions", type="evaluation")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"preds": preds, "log_probs": log_probs, "labels": labels}, f.name)
            tmp_path = f.name
        preds_artifact.add_file(tmp_path, name="test_predictions.pt")
        run.log_artifact(preds_artifact)
        os.unlink(tmp_path)

    return {"accuracy": accuracy, "nll": nll}


def evaluate_regressor(model, test_loader: DataLoader, test_y: torch.Tensor, run=None) -> dict:
    """
    Evaluate a TwoLayerDSPP regression model on a test set.

    Fixes the missing evaluation cell from the original dspp.ipynb.

    Args:
        model:       TwoLayerDSPP (must be in eval mode)
        test_loader: DataLoader yielding (x_batch, y_batch)
        test_y:      Ground-truth target tensor (N,) on any device
        run:         Optional W&B run object. If provided, logs eval metrics and
                     saves a predictions artifact.

    Returns:
        {'rmse': float, 'nll': float}
        rmse is on the normalised label scale; nll is in nats.
    """
    model.eval()
    mus, variances, lls = model.predict(test_loader)

    test_y_cpu = test_y.cpu()
    rmse = (mus - test_y_cpu).pow(2).mean().sqrt().item()
    nll = -lls.mean().item()

    if run is not None:
        import wandb
        run.log({"eval/rmse": rmse, "eval/nll": nll})
        preds_artifact = wandb.Artifact("predictions", type="evaluation")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"mus": mus, "variances": variances, "lls": lls}, f.name)
            tmp_path = f.name
        preds_artifact.add_file(tmp_path, name="test_predictions.pt")
        run.log_artifact(preds_artifact)
        os.unlink(tmp_path)

    return {"rmse": rmse, "nll": nll}
