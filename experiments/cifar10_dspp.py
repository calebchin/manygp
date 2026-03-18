"""
CIFAR-10 classification with a two-layer DSPP + CNN feature extractor.

Usage:
    python experiments/cifar10_dspp.py --config configs/cifar10_dspp.yaml
"""

import argparse
import os
import sys
import tempfile

import torch
import yaml
from gpytorch.mlls import DeepPredictiveLogLikelihood

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.models.feature_extractors import CNNFeatureExtractor
from src.models.classifiers import TwoLayerDSPPClassifier
from src.training.trainer import (
    extract_cnn_features,
    init_inducing_points_kmeans,
    pretrain_cnn,
    train_dspp,
)
from src.training.evaluate import evaluate_classifier


def main(cfg: dict, config_path: str) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B init ──────────────────────────────────────────────────────────────
    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        run = wandb.init(
            project=wandb_cfg.get("project", "manygp"),
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("run_name") or None,
            config=cfg,
        )
        config_artifact = wandb.Artifact("config", type="config")
        config_artifact.add_file(config_path)
        run.log_artifact(config_artifact)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader, dataset_train, dataset_test = get_cifar10_loaders(
        data_root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(dataset_train)}, test size: {len(dataset_test)}")

    # ── CNN pretraining ───────────────────────────────────────────────────────
    cnn_cfg = cfg["cnn"]
    cnn = CNNFeatureExtractor(
        latent_dim=cnn_cfg["latent_dim"],
        num_classes=cnn_cfg["num_classes"],
    ).to(device)

    pretrain_epochs = 1 if smoke_test else cnn_cfg["pretrain_epochs"]
    pretrain_cnn(
        cnn=cnn,
        train_loader=train_loader,
        epochs=pretrain_epochs,
        device=device,
        lr=cnn_cfg["pretrain_lr"],
        milestones=cnn_cfg["pretrain_milestones"],
        run=run,
    )

    # Evaluate CNN standalone accuracy
    cnn.eval()
    correct = total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = cnn.forward_cls(x_batch.to(device)).argmax(dim=-1).cpu()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    cnn_accuracy = correct / total
    print(f"CNN standalone test accuracy: {cnn_accuracy * 100:.2f}%")
    if run is not None:
        run.log({"pretrain/cnn_accuracy": cnn_accuracy})

    # ── Inducing point initialisation ─────────────────────────────────────────
    dspp_cfg = cfg["dspp"]
    pool = extract_cnn_features(
        cnn=cnn,
        loader=train_loader,
        n_samples=dspp_cfg["inducing_pool_size"],
        device=device,
    )
    inducing = init_inducing_points_kmeans(pool, dspp_cfg["num_inducing_pts"]).to(device)
    print(f"Inducing points shape: {inducing.shape}")

    # ── DSPP model ────────────────────────────────────────────────────────────
    dspp = TwoLayerDSPPClassifier(
        latent_dim=cnn_cfg["latent_dim"],
        hidden_dim=dspp_cfg["hidden_dim"],
        num_classes=cnn_cfg["num_classes"],
        inducing_points=inducing,
        num_inducing=dspp_cfg["num_inducing_pts"],
        Q=dspp_cfg["num_quad_sites"],
    ).to(device)

    train_cfg = cfg["training"]
    objective = DeepPredictiveLogLikelihood(
        dspp.likelihood,
        dspp,
        num_data=len(dataset_train),
        beta=train_cfg["beta_reg"],
    )

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs = 1 if smoke_test else train_cfg["num_epochs"]
    train_dspp(
        model=dspp,
        objective=objective,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=train_cfg["initial_lr"],
        milestones=train_cfg["milestones"],
        device=device,
        cnn=cnn,
        run=run,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate_classifier(dspp, test_loader, cnn=cnn, device=device, run=run)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test NLL:      {metrics['nll']:.4f} nats")

    # ── Model checkpoint artifact ─────────────────────────────────────────────
    if run is not None:
        import wandb
        ckpt_artifact = wandb.Artifact("model", type="model")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"dspp": dspp.state_dict(), "cnn": cnn.state_dict()}, f.name)
            tmp_path = f.name
        ckpt_artifact.add_file(tmp_path, name="checkpoint.pt")
        run.log_artifact(ckpt_artifact)
        os.unlink(tmp_path)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 DSPP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, config_path=args.config)
