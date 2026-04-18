"""
experiments/uncertainty_trajectory.py

Uncertainty trajectory analysis: how SNGP outputs evolve as inputs become
progressively harder to classify.

Given:
  • A checkpoint from any of the 8 SNGP experiments
  • X — FloatTensor of shape (N, T, C, H, W):
        N series, each of T distortion levels (clean → hardest)
  • y — LongTensor of shape (N,): true class index per series

For each (n, t) computes — all indexed at the TRUE CLASS c = y[n]:

  f_c(x*)       raw GP mean logit at true class           scalar
  σ²_c(x*)      GP posterior variance at true class       scalar
                (σ² is actually (batch, num_classes); we take index c)
  p̂_c det       softmax(f / √(1 + π/8 · σ²))_c          scalar  [deterministic]
  p̂_c mc        E[softmax(f̃)]_c, f̃~N(f, diag(σ²))       scalar  [MC, S=50]

W&B layout — 1 run = 1 checkpoint:
  series_NNN/
    f_c          — f_c(x*) vs distortion level t
    sigma_sq_c   — σ²_c(x*) vs distortion level t
    prob_det_c   — p̂_c deterministic vs t
    prob_mc_c    — p̂_c MC (S=50) vs t

Usage:
    python experiments/uncertainty_trajectory.py \\
        --experiment  hybrid_sngp \\
        --config      configs/experiment_april4_hybrid_sngp.yaml \\
        --checkpoint  checkpoints_april4/hybrid_sngp/seed0/.../best_model.pt \\
        --data-x      data/trajectory_X.pt \\
        --data-y      data/trajectory_y.pt \\
        --run-name    traj_hybrid_sngp_seed0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Model construction ────────────────────────────────────────────────────────

def build_model(exp_type: str, cfg: dict, device: torch.device):
    model_cfg = cfg["model"]

    if exp_type in ("hybrid_sngp", "ms_sngp_sn", "sngp", "sngp_augmented"):
        from src.models.sngp import SNGPResNetClassifier
        return SNGPResNetClassifier(
            num_classes     = model_cfg["num_classes"],
            widen_factor    = model_cfg.get("widen_factor", 10),
            hidden_dim      = model_cfg["hidden_dim"],
            spec_norm_bound = model_cfg["spec_norm_bound"],
            num_inducing    = model_cfg["num_inducing"],
            ridge_penalty   = model_cfg["ridge_penalty"],
            feature_scale   = model_cfg["feature_scale"],
            gp_cov_momentum = model_cfg["gp_cov_momentum"],
            normalize_input = model_cfg["normalize_input"],
        ).to(device)

    elif exp_type == "ms_sngp_no_skip":
        from src.models.sngp import WRNNoSkipSupConSNGPClassifier
        return WRNNoSkipSupConSNGPClassifier(
            embedding_dim       = model_cfg["embedding_dim"],
            num_classes         = model_cfg["num_classes"],
            widen_factor        = model_cfg.get("widen_factor", 10),
            hidden_dims         = model_cfg["hidden_dims"],
            dropout_rate        = model_cfg["dropout_rate"],
            num_inducing        = model_cfg["num_inducing"],
            ridge_penalty       = model_cfg["ridge_penalty"],
            feature_scale       = model_cfg["feature_scale"],
            gp_cov_momentum     = model_cfg["gp_cov_momentum"],
            normalize_input     = model_cfg["normalize_input"],
            kernel_type         = model_cfg.get("kernel_type", "normalized_rbf"),
            input_normalization = model_cfg.get("input_normalization", "l2"),
            kernel_scale        = model_cfg.get("kernel_scale", 1.0),
            length_scale        = model_cfg.get("length_scale", 1.0),
        ).to(device)

    elif exp_type in ("ms_sngp", "supcon_sngp"):
        from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier
        return CifarResNetSupConSNGPClassifier(
            embedding_dim       = model_cfg["embedding_dim"],
            num_classes         = model_cfg["num_classes"],
            widen_factor        = model_cfg.get("widen_factor", 10),
            hidden_dims         = model_cfg.get("hidden_dims", []),
            dropout_rate        = model_cfg.get("dropout_rate", 0.0),
            num_inducing        = model_cfg["num_inducing"],
            ridge_penalty       = model_cfg["ridge_penalty"],
            feature_scale       = model_cfg["feature_scale"],
            gp_cov_momentum     = model_cfg["gp_cov_momentum"],
            normalize_input     = model_cfg.get("normalize_input", False),
            kernel_type         = model_cfg.get("kernel_type", "legacy"),
            input_normalization = model_cfg.get("input_normalization", None),
            kernel_scale        = model_cfg.get("kernel_scale", 1.0),
            length_scale        = model_cfg.get("length_scale", 1.0),
        ).to(device)

    else:
        raise ValueError(
            f"Unknown experiment type: {exp_type!r}. "
            "Supported: hybrid_sngp | ms_sngp | ms_sngp_sn | ms_sngp_no_skip | "
            "sngp | sngp_augmented | supcon_sngp"
        )


# ── Forward pass ──────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_sngp(
    model,
    x: torch.Tensor,       # (N, C, H, W)
    device: torch.device,
    num_mc: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (all CPU, all shape (N, num_classes)):
        logits    — f(x*), raw GP mean
        variances — σ²_c(x*) per class; shape (N, num_classes)
                    confirmed from einsum "bi,kij,bj->bk" in the GP layer,
                    where precision_matrix is (num_classes, num_inducing, num_inducing)
        probs_mc  — MC Laplace: mean_S softmax(f + √σ² · ε),  ε~N(0,I)
        probs_det — deterministic rescaling: softmax(f / √(1 + π/8·σ²))
                    each logit rescaled by its own per-class σ²
    """
    from src.models.sngp import laplace_predictive_probs

    x = x.to(device)
    logits, variances = model(x, return_cov=True)
    # logits, variances: (N, num_classes)

    probs_mc = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc)

    # Deterministic Laplace: per-class rescaling
    scale     = torch.sqrt(1.0 + (math.pi / 8.0) * variances)  # (N, num_classes)
    probs_det = (logits / scale).softmax(dim=-1)                 # (N, num_classes)

    return logits.cpu(), variances.cpu(), probs_mc.cpu(), probs_det.cpu()


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_series(
    run,
    n: int,
    true_label: int,
    f_seq: list[float],
    sigma_seq: list[float],
    prob_det_seq: list[float],
    prob_mc_seq: list[float],
    label_names: list[str],
) -> None:
    """
    Log one series as exactly 4 line charts under section series_NNN/.

    All four quantities are indexed at the TRUE CLASS c = true_label.
    """
    import wandb

    T       = len(f_seq)
    cls     = label_names[true_label]
    prefix  = f"series_{n:03d}"

    table = wandb.Table(
        columns=["t", "f_c", "sigma_sq_c", "prob_det_c", "prob_mc_c"],
        data=[[t, f_seq[t], sigma_seq[t], prob_det_seq[t], prob_mc_seq[t]]
              for t in range(T)],
    )

    run.log({
        # 1. f_c(x*) vs t
        f"{prefix}/f_c": wandb.plot.line(
            table, "t", "f_c",
            title=f"[Series {n} | class={cls}]  f_c(x*)  vs distortion level",
        ),
        # 2. σ²_c(x*) vs t
        f"{prefix}/sigma_sq_c": wandb.plot.line(
            table, "t", "sigma_sq_c",
            title=f"[Series {n} | class={cls}]  σ²_c(x*)  vs distortion level",
        ),
        # 3. p̂_c deterministic vs t
        f"{prefix}/prob_det_c": wandb.plot.line(
            table, "t", "prob_det_c",
            title=f"[Series {n} | class={cls}]  p̂_c deterministic  softmax(f/√(1+π/8·σ²))  vs t",
        ),
        # 4. p̂_c MC (S=50) vs t
        f"{prefix}/prob_mc_c": wandb.plot.line(
            table, "t", "prob_mc_c",
            title=f"[Series {n} | class={cls}]  p̂_c MC (S=50)  E[softmax(f̃)]  vs t",
        ),
        # raw table for download / inspection
        f"{prefix}/data": table,
    })


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tensor(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".npy":
        return torch.from_numpy(np.load(str(p)))
    return torch.load(str(p), weights_only=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uncertainty trajectory analysis for SNGP models"
    )
    parser.add_argument("--experiment", required=True,
                        help="hybrid_sngp | ms_sngp | ms_sngp_sn | ms_sngp_no_skip | "
                             "sngp | sngp_augmented | supcon_sngp")
    parser.add_argument("--config",     required=True,
                        help="YAML config used during training")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--data-x",    required=True, dest="data_x",
                        help="X tensor (.pt or .npy), shape (N, T, C, H, W)")
    parser.add_argument("--data-y",    required=True, dest="data_y",
                        help="y tensor (.pt or .npy), shape (N,) — true class per series")
    parser.add_argument("--num-mc",    type=int, default=50, dest="num_mc",
                        help="MC samples for Laplace approximation (default 50)")
    parser.add_argument("--run-name",  default=None, dest="run_name")
    parser.add_argument("--label-names", default=None, dest="label_names",
                        help="Comma-separated class names (default: CIFAR-10)")
    parser.add_argument("--wandb-project", default="april_4_experiments",
                        dest="wandb_project")
    parser.add_argument("--wandb-entity",  default="sta414manygp",
                        dest="wandb_entity")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Building {args.experiment} ...")
    model = build_model(args.experiment, cfg, device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch   = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_accuracy", ckpt.get("val_acc", "?"))
    print(f"Loaded — epoch {epoch}, val_acc={val_acc}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X = load_tensor(args.data_x).float()   # (N, T, C, H, W)
    y = load_tensor(args.data_y).long()    # (N,)

    assert X.ndim == 5, f"Expected (N,T,C,H,W), got {X.shape}"
    assert y.ndim == 1 and y.shape[0] == X.shape[0]

    N, T, C, H, W = X.shape
    print(f"Data: N={N} series, T={T} levels, image=({C},{H},{W})")

    label_names = args.label_names.split(",") if args.label_names else CIFAR10_CLASSES

    # ── W&B ───────────────────────────────────────────────────────────────────
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name or f"traj_{args.experiment}_ep{epoch}",
        config=dict(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            checkpoint_epoch=epoch,
            val_accuracy=val_acc,
            N_series=N,
            T_levels=T,
            num_mc=args.num_mc,
        ),
        tags=["uncertainty_trajectory"],
    )
    print(f"W&B: {run.url}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    # All quantities indexed at true class c = y[n]
    all_f        = [[0.0] * T for _ in range(N)]
    all_sigma    = [[0.0] * T for _ in range(N)]
    all_prob_det = [[0.0] * T for _ in range(N)]
    all_prob_mc  = [[0.0] * T for _ in range(N)]

    true_indices = y.numpy()

    print("Running inference ...")
    for t in range(T):
        X_t = X[:, t]   # (N, C, H, W)
        f_t, var_t, prob_mc_t, prob_det_t = forward_sngp(model, X_t, device, args.num_mc)
        # All (N, num_classes); index at true class below

        for n in range(N):
            c = true_indices[n]
            all_f[n][t]        = f_t[n, c].item()
            all_sigma[n][t]    = var_t[n, c].item()    # σ²_c: variance at true class
            all_prob_mc[n][t]  = prob_mc_t[n, c].item()
            all_prob_det[n][t] = prob_det_t[n, c].item()

        # Progress
        idx = np.arange(N)
        print(f"  t={t:3d}/{T-1}  "
              f"f_c={f_t[idx, true_indices].mean():.3f}  "
              f"σ²_c={var_t[idx, true_indices].mean():.5f}  "
              f"p̂_mc={prob_mc_t[idx, true_indices].mean():.4f}  "
              f"p̂_det={prob_det_t[idx, true_indices].mean():.4f}")

    # ── Log to W&B ────────────────────────────────────────────────────────────
    print("\nLogging to W&B ...")
    for n in range(N):
        log_series(
            run         = run,
            n           = n,
            true_label  = y[n].item(),
            f_seq       = all_f[n],
            sigma_seq   = all_sigma[n],
            prob_det_seq= all_prob_det[n],
            prob_mc_seq = all_prob_mc[n],
            label_names = label_names,
        )
    print(f"  {N} series logged.")

    run.finish()
    print("Done.")


if __name__ == "__main__":
    main()
