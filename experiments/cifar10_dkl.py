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
from src.models.dkl import GP, DKLModel, initial_values
from src.training.trainer import (
    extract_cnn_features,
    init_inducing_points_kmeans,
    train_dkl,
)
from src.training.evaluate import evaluate_classifier

def train_dkl(
    model,
    objective,
    train_loader,
    num_epochs: int,
    lr: float,
    milestones: List[int],
    device,
    run=None,
) -> List[float]:
    """
    Train a DSPP model with Adam + MultiStepLR.

    Fixes the tqdm.notebook crash from the original dspp.ipynb by using
    tqdm.auto instead.

    Args:
        model:        DSPP model (TwoLayerDSPPClassifier or TwoLayerDSPP).
        objective:    gpytorch Likelihood instance.
        train_loader: DataLoader yielding (x_batch, y_batch).
        num_epochs:   Number of training epochs.
        lr:           Initial learning rate.
        milestones:   Epoch milestones for MultiStepLR (gamma=0.1).
        device:       torch.device
        cnn:          Optional feature extractor (CIFAR-10 only).
                      If provided, kept in eval mode; features are extracted in-loop.
        run:          Optional W&B run object. If provided, logs per-epoch metrics.

    Returns:
        List of per-epoch average losses.
    """
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 5e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': objective.parameters()},
    ], lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    mll = VariationalELBO(objective, model.gp_layer, num_data=len(train_loader.dataset))

    epoch_losses = []
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for epoch in epochs_iter:
        model.train()
        objective.train()
        total_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc="Minibatch", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        epochs_iter.set_postfix(loss=avg_loss)

        print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")

        if run is not None:
            run.log({
                "train/loss": avg_loss,
                "train/epoch": epoch + 1,
                "train/lr": scheduler.get_last_lr()[0],
            })

    return epoch_losses


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

    # ── CNN initialiation ───────────────────────────────────────────────────────
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
    # pool = extract_cnn_features(
    #     cnn=cnn,
    #     loader=train_loader,
    #     n_samples=dkl_cfg["inducing_pool_size"],
    #     device=device,
    # )
    # inducing = init_inducing_points_kmeans(pool, dkl_cfg["num_inducing_pts"]).to(device)
    initial_inducing_points, initial_lengthscale = initial_values(
        dataset_train, cnn, dkl_cfg["num_inducing_pts"]
    )
    print(f"Inducing points shape: {initial_inducing_points.shape}")

    # ── DKL model ────────────────────────────────────────────────────────────
    dp_num_output = inducing.shape[1] if dkl_cfg["per_feature"] else dkl_cfg["num_output"]
    gp = GP(
        num_outputs=dp_num_output,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=dkl_cfg.get("kernel", "RBF"),
    ).to(device)
    
    objective = SoftmaxLikelihood(num_classes=cnn_cfg["num_classes"], mixing_weights=False).to(device)
    dkl = DKLModel(cnn, gp, objective, per_feature=dkl_cfg["per_feature"]).to(device)

    train_cfg = cfg["training"]

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
