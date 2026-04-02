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
from gpytorch.likelihoods import SoftmaxLikelihood

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.models.due.wide_resnet import WideResNet
from src.models.feature_extractors import CNNFeatureExtractor
from src.models.dkl import GP, DKLModel
from src.training.trainer import (
    extract_cnn_features,
    init_inducing_points_kmeans,
    train_dkl,
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
    cnn = WideResNet(
        input_size=32,
        spectral_conv=cnn_cfg["spectral_conv"],
        spectral_bn=cnn_cfg["spectral_bn"]
    ).to(device)
    # cnn = CNNFeatureExtractor(
    #     latent_dim=cnn_cfg["latent_dim"],
    #     num_classes=cnn_cfg["num_classes"],
    # ).to(device)

    # ── Inducing point initialisation ─────────────────────────────────────────
    dkl_cfg = cfg["dkl"]
    pool = extract_cnn_features(
        cnn=cnn,
        loader=train_loader,
        n_samples=dkl_cfg["inducing_pool_size"],
        device=device,
    )
    inducing = init_inducing_points_kmeans(pool, dkl_cfg["num_inducing_pts"]).to(device)
    print(f"Inducing points shape: {inducing.shape}")

    # ── DSPP model ────────────────────────────────────────────────────────────
    gp = GP(
        inducing_points=inducing,
        num_inducing=dkl_cfg["num_inducing_pts"],
        num_output=dkl_cfg["num_output"],
        per_feature=dkl_cfg["per_feature"]
    ).to(device)
    dkl = DKLModel(cnn, gp, per_feature=dkl_cfg["per_feature"]).to(device)

    train_cfg = cfg["training"]
    objective = SoftmaxLikelihood(num_features=dkl_cfg["num_output"], num_classes=cnn_cfg["num_classes"]).to(device)

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs = 1 if smoke_test else train_cfg["num_epochs"]
    train_dkl(
        model=dkl,
        objective=objective,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=train_cfg["initial_lr"],
        milestones=train_cfg["milestones"],
        device=device,
        run=run,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    # TODO: Add evaluation for DKL
    metrics = evaluate_classifier(dkl, test_loader, cnn=cnn, device=device, run=run)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test NLL:      {metrics['nll']:.4f} nats")

    # ── Model checkpoint artifact ─────────────────────────────────────────────
    if run is not None:
        import wandb
        ckpt_artifact = wandb.Artifact("model", type="model")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"dkl": dkl.state_dict()}, f.name)
            tmp_path = f.name
        ckpt_artifact.add_file(tmp_path, name="checkpoint.pt")
        run.log_artifact(ckpt_artifact)
        os.unlink(tmp_path)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 DKL experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, config_path=args.config)
