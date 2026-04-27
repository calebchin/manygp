"""
UCI Bike Sharing regression with a two-layer DSPP.

Performs a grid search over beta_reg and hidden_dim using a train/val/test
split (15:3:2 ratio), then retrains the best config on train+val and evaluates
on the held-out test set.

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
from torch.utils.data import DataLoader, TensorDataset

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.bike import get_bike_loaders
from src.models.regressors import TwoLayerDSPP
from src.training.trainer import init_inducing_points_kmeans, train_dspp
from src.training.evaluate import evaluate_regressor

_BETA_GRID = [0.01, 0.05, 0.2, 1.0]
_DIM_GRID = [3, 5]


def _build_and_train(
    train_x, train_y, train_loader,
    input_dims, hidden_dim, beta_reg,
    dspp_cfg, train_cfg, num_epochs,
    device, run, run_tag,
):
    """Initialise model, train, and return (model, val_metrics)."""
    inducing = init_inducing_points_kmeans(
        train_x.cpu(), dspp_cfg["num_inducing_pts"]
    ).to(device)

    model = TwoLayerDSPP(
        input_dims=input_dims,
        inducing_points=inducing,
        num_inducing=dspp_cfg["num_inducing_pts"],
        hidden_dim=hidden_dim,
        Q=dspp_cfg["num_quad_sites"],
    ).to(device)

    objective = DeepPredictiveLogLikelihood(
        model.likelihood,
        model,
        num_data=train_x.size(0),
        beta=beta_reg,
    )

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

    return model


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
        train_loader, val_loader, test_loader,
        train_x, train_y, val_x, val_y, test_x, test_y,
        train_n,
    ) = get_bike_loaders(
        data_path=data_cfg["path"],
        batch_size=data_cfg["batch_size"],
        train_frac=data_cfg["train_frac"],
        val_frac=data_cfg.get("val_frac", 0.15),
        smoke_test=smoke_test,
        device=device_str,
    )
    print(
        f"Train: {train_n}, val: {val_x.size(0)}, test: {test_x.size(0)}, "
        f"input_dims: {train_x.size(1)}"
    )

    dspp_cfg = cfg["dspp"]
    train_cfg = cfg["training"]
    num_epochs = 1 if smoke_test else train_cfg["num_epochs"]
    input_dims = train_x.size(1)

    # ── Grid search over val set ───────────────────────────────────────────────
    best_val_nll = float("inf")
    best_beta = None
    best_dim = None

    for hidden_dim in _DIM_GRID:
        for beta_reg in _BETA_GRID:
            run_tag = f"beta_{beta_reg}_dim{hidden_dim}"
            print(f"\n── Grid search: {run_tag} ──")

            model = _build_and_train(
                train_x, train_y, train_loader,
                input_dims, hidden_dim, beta_reg,
                dspp_cfg, train_cfg, num_epochs,
                device, run=None, run_tag=run_tag,
            )

            val_metrics = evaluate_regressor(
                model, val_loader, val_y, dataset_name="bike/dspp", run=None
            )
            val_nll = val_metrics["nll"]
            print(f"  Val NLL: {val_nll:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")

            if run is not None:
                run.log({
                    f"bike/dspp/hparam/{run_tag}/val_nll": val_nll,
                    f"bike/dspp/hparam/{run_tag}/val_rmse": val_metrics["rmse"],
                })

            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_beta = beta_reg
                best_dim = hidden_dim

    print(f"\nBest config: beta_reg={best_beta}, hidden_dim={best_dim} (val NLL={best_val_nll:.4f})")
    if run is not None:
        run.log({
            "bike/dspp/best/beta_reg": best_beta,
            "bike/dspp/best/hidden_dim": best_dim,
            "bike/dspp/best/val_nll": best_val_nll,
        })

    # ── Retrain best config on train + val ────────────────────────────────────
    print("\n── Retraining best config on train+val ──")
    trainval_x = torch.cat([train_x, val_x], dim=0)
    trainval_y = torch.cat([train_y, val_y], dim=0)
    trainval_loader = DataLoader(
        TensorDataset(trainval_x, trainval_y),
        batch_size=data_cfg["batch_size"],
        shuffle=True,
    )

    final_model = _build_and_train(
        trainval_x, trainval_y, trainval_loader,
        input_dims, best_dim, best_beta,
        dspp_cfg, train_cfg, num_epochs,
        device, run=run, run_tag="final",
    )

    # ── Final evaluation on test set ──────────────────────────────────────────
    metrics = evaluate_regressor(
        final_model, test_loader, test_y, dataset_name="bike/dspp", run=run
    )
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test NLL:  {metrics['nll']:.4f} nats")

    # ── Model checkpoint artifact ─────────────────────────────────────────────
    if run is not None:
        import wandb
        ckpt_artifact = wandb.Artifact("model", type="model")
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model": final_model.state_dict()}, f.name)
            tmp_path = f.name
        ckpt_artifact.add_file(tmp_path, name="checkpoint.pt")
        run.log_artifact(ckpt_artifact)
        os.unlink(tmp_path)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bike sharing DSPP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", default=None, help="W&B run name (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run_name is not None:
        cfg.setdefault("wandb", {})["run_name"] = args.run_name

    main(cfg, config_path=args.config)
