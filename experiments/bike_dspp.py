"""
UCI Bike Sharing regression with a two-layer DSPP.

Fixes from the original dspp.ipynb:
  - train_loader was never created (now returned by get_bike_loaders)
  - tqdm.notebook crash (trainer uses tqdm.auto)
  - No evaluation cell existed (evaluate_regressor added)

Usage:
    python experiments/bike_dspp.py --config configs/bike_dspp.yaml
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

from src.data.bike import get_bike_loaders
from src.models.regressors import TwoLayerDSPP
from src.training.trainer import init_inducing_points_kmeans, train_dspp
from src.training.evaluate import evaluate_regressor


def main(cfg: dict, config_path: str) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
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
    data_cfg = cfg["data"]
    (
        train_loader, test_loader,
        train_x, train_y, test_x, test_y,
        train_n,
    ) = get_bike_loaders(
        data_path=data_cfg["path"],
        batch_size=data_cfg["batch_size"],
        train_frac=data_cfg["train_frac"],
        smoke_test=smoke_test,
        device=device_str,
    )
    print(f"Train size: {train_n}, test size: {test_x.size(0)}, input_dims: {train_x.size(1)}")

    # ── Inducing point initialisation ─────────────────────────────────────────
    dspp_cfg = cfg["dspp"]
    inducing = init_inducing_points_kmeans(
        train_x.cpu(), dspp_cfg["num_inducing_pts"]
    ).to(device)
    print(f"Inducing points shape: {inducing.shape}")

    # ── DSPP model ────────────────────────────────────────────────────────────
    model = TwoLayerDSPP(
        input_dims=train_x.size(1),
        inducing_points=inducing,
        num_inducing=dspp_cfg["num_inducing_pts"],
        hidden_dim=dspp_cfg["hidden_dim"],
        Q=dspp_cfg["num_quad_sites"],
    ).to(device)

    train_cfg = cfg["training"]
    objective = DeepPredictiveLogLikelihood(
        model.likelihood,
        model,
        num_data=train_n,
        beta=train_cfg["beta_reg"],
    )

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs = 1 if smoke_test else train_cfg["num_epochs"]
    train_dspp(
        model=model,
        objective=objective,
        train_loader=train_loader,
        num_epochs=num_epochs,
        lr=train_cfg["initial_lr"],
        milestones=train_cfg["milestones"],
        device=device,
        run=run,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate_regressor(model, test_loader, test_y, run=run)
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test NLL:  {metrics['nll']:.4f} nats")

    # ── Model checkpoint artifact ─────────────────────────────────────────────
    if run is not None:
        import wandb
        ckpt_artifact = wandb.Artifact("model", type="model")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": model.state_dict()}, f.name)
            tmp_path = f.name
        ckpt_artifact.add_file(tmp_path, name="checkpoint.pt")
        run.log_artifact(ckpt_artifact)
        os.unlink(tmp_path)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bike sharing DSPP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, config_path=args.config)
