"""
CIFAR-10 Multi-Similarity Loss + SNGP classification training.

Identical architecture to SupCon+SNGP (CifarResNetSupConSNGPClassifier), but
replaces the SupCon loss with Multi-Similarity Loss (Wang et al., CVPR 2019).

Why MS Loss instead of SupCon:
  SupCon gives uniform gradient weight to every same-class pair.  MS Loss
  reweights pairs by how informative they are — pairs straddling the decision
  boundary get high weight, already-correct pairs get near-zero weight.  This
  produces tighter, more distance-calibrated clusters, which directly improves
  the GP's ability to map distance → uncertainty.

The MS Loss operates on the (B*V, D) flattened GP-input embeddings (the
features that feed into the random-feature GP layer), exactly where distance
structure matters most for uncertainty.

After training, runs OOD detection (SVHN, CIFAR-100) and CIFAR-10-C
corruption robustness evaluation in the same W&B run.

Usage:
    python experiments/cifar10_ms_sngp.py \
        --config configs/experiment_april4_ms_sngp.yaml --seed 0
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.sngp import laplace_predictive_probs
from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier
from src.training.contrastive import MultiSimilarityLoss
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper
from src.utils.model_summary import print_model_summary


def resolve_timestamped_checkpoint_path(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(p.parent / ts / p.name)


def update_topk_checkpoints(
    saved_checkpoints: list[dict],
    top_k: int,
    checkpoint_path: str,
    state: dict,
    metric_name: str,
    metric_value: float,
    epoch: int,
    lower_is_better: bool = False,
) -> None:
    if top_k <= 0:
        return
    p = Path(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    candidate = p.parent / f"{p.stem}_epoch{epoch:03d}_{metric_name}{metric_value:.4f}{p.suffix or '.pt'}"

    if len(saved_checkpoints) < top_k:
        torch.save(state, candidate)
        saved_checkpoints.append({"metric": metric_value, "path": candidate, "epoch": epoch})
    else:
        worst = (max if lower_is_better else min)(
            saved_checkpoints, key=lambda x: (x["metric"], -x["epoch"])
        )
        if (lower_is_better and metric_value >= worst["metric"]) or \
           (not lower_is_better and metric_value <= worst["metric"]):
            return
        torch.save(state, candidate)
        saved_checkpoints.append({"metric": metric_value, "path": candidate, "epoch": epoch})
        worst["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst)

    saved_checkpoints.sort(key=lambda x: x["metric"], reverse=not lower_is_better)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    ms_loss_fn: MultiSimilarityLoss,
    ce_loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    ms_weight: float,
    ce_weight: float,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    model.train()
    running_total = running_ms = running_ce = 0.0
    total_correct = total_examples = 0

    progress = tqdm(loader, desc=f"MS-SNGP Epoch {epoch}", leave=False)
    for views, labels in progress:
        labels = labels.to(device, non_blocking=True)
        B, V, C, H, W = views.shape
        views = views.to(device, non_blocking=True).view(B * V, C, H, W)
        ce_labels = labels.repeat_interleave(V)   # (B*V,)

        optimizer.zero_grad(set_to_none=True)
        logits, gp_inputs = model(views, update_precision=True, return_features=True)
        # gp_inputs: (B*V, D) — the embeddings feeding into the GP layer
        # MS Loss operates on these directly (flattened, no view dimension needed)
        ms_loss = ms_loss_fn(gp_inputs, ce_labels)
        ce_loss = ce_loss_fn(logits, ce_labels)
        total_loss = ms_weight * ms_loss + ce_weight * ce_loss
        total_loss.backward()
        optimizer.step()

        global_step += 1
        running_total += total_loss.item()
        running_ms    += ms_loss.item()
        running_ce    += ce_loss.item()
        total_correct   += (logits.argmax(dim=1) == ce_labels).sum().item()
        total_examples  += ce_labels.size(0)
        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            ms=f"{ms_loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
        )

        if run is not None and log_every_steps and global_step % log_every_steps == 0:
            run.log({
                "train/step_total_loss": total_loss.item(),
                "train/step_ms_loss":    ms_loss.item(),
                "train/step_ce_loss":    ce_loss.item(),
                "train/global_step":     global_step,
                "train/epoch":           epoch,
                "train/lr_step":         optimizer.param_groups[0]["lr"],
            })

    return {
        "total_loss": running_total / len(loader),
        "ms_loss":    running_ms    / len(loader),
        "ce_loss":    running_ce    / len(loader),
        "accuracy":   total_correct / total_examples,
    }, global_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_mc_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    running_loss = total_correct = total_examples = 0
    total_nll = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, variances = model(images, return_cov=True)
        probs    = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        running_loss    += (-log_probs.gather(1, labels.unsqueeze(1)).mean()).item()
        total_correct   += (probs.argmax(dim=1) == labels).sum().item()
        total_examples  += labels.size(0)
        total_nll       += (-log_probs.gather(1, labels.unsqueeze(1)).sum()).item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t  = torch.cat(all_probs)
    all_labels_t = torch.cat(all_labels)
    return {
        "loss":     running_loss   / len(loader),
        "accuracy": total_correct  / total_examples,
        "nll":      total_nll      / total_examples,
        "ece":      _classification_ece(all_probs_t, all_labels_t),
    }


def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = cfg.get("training", {}).get("seed")
    if seed is not None:
        import random, numpy as np
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        run = wandb.init(
            project=wandb_cfg.get("project", "sngp"),
            entity=wandb_cfg.get("entity") or "sta414manygp",
            name=wandb_cfg.get("run_name") or None,
            config=cfg,
        )

    data_cfg = cfg["data"]
    train_loader, _, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        get_cifar10_supcon_loaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            smoke_test=smoke_test,
        )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    model_cfg = cfg["model"]
    model = CifarResNetSupConSNGPClassifier(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dims=model_cfg["hidden_dims"],
        dropout_rate=model_cfg["dropout_rate"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
        kernel_type=model_cfg["kernel_type"],
        input_normalization=model_cfg["input_normalization"],
        kernel_scale=model_cfg["kernel_scale"],
        length_scale=model_cfg["length_scale"],
    ).to(device)
    print_model_summary(model)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1 if smoke_test else train_cfg["epochs"]
    )

    ms_loss_fn = MultiSimilarityLoss(
        alpha=train_cfg.get("ms_alpha", 2.0),
        beta=train_cfg.get("ms_beta",  50.0),
        base=train_cfg.get("ms_base",  0.5),
        eps=train_cfg.get("ms_eps",    0.1),
    )
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    ms_weight  = train_cfg.get("ms_loss_weight", 1.0)
    ce_weight  = train_cfg.get("ce_loss_weight",  1.0)
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval   = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps")
    num_mc_samples  = train_cfg.get("num_mc_samples", 10)

    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    resolved_ckpt   = resolve_timestamped_checkpoint_path(checkpoint_path) if checkpoint_path else None
    top_k            = output_cfg.get("top_k", 5)
    checkpoint_metric = output_cfg.get("checkpoint_metric", "val_accuracy")
    lower_is_better   = checkpoint_metric in ("val_ece", "val_loss")
    saved_checkpoints: list[dict] = []
    global_step = 0
    best_val_acc = -1.0

    model.reset_precision_matrix()
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_ckpt

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_metrics, global_step = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            ms_loss_fn=ms_loss_fn, ce_loss_fn=ce_loss_fn, device=device,
            epoch=epoch, ms_weight=ms_weight, ce_weight=ce_weight,
            run=run, log_every_steps=log_every_steps, global_step=global_step,
        )
        scheduler.step()

        val_metrics = None
        if epoch % eval_interval == 0 or epoch == num_epochs:
            val_metrics = evaluate(model, val_loader, device, num_mc_samples)
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {train_metrics['total_loss']:.4f} | "
                f"MS: {train_metrics['ms_loss']:.4f} | "
                f"CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"Val NLL: {val_metrics['nll']:.4f} | "
                f"Val ECE: {val_metrics['ece']:.4f}"
            )
            epoch_progress.set_postfix(
                val_acc=f"{val_metrics['accuracy']*100:.2f}%",
                ms=f"{train_metrics['ms_loss']:.4f}",
            )
        else:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Total: {train_metrics['total_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}%"
            )

        if run is not None:
            log_data = {
                "train/total_loss": train_metrics["total_loss"],
                "train/ms_loss":    train_metrics["ms_loss"],
                "train/ce_loss":    train_metrics["ce_loss"],
                "train/accuracy":   train_metrics["accuracy"],
                "train/lr":         optimizer.param_groups[0]["lr"],
                "train/epoch":      epoch,
            }
            if val_metrics:
                log_data.update({
                    "val/loss":     val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/nll":      val_metrics["nll"],
                    "val/ece":      val_metrics["ece"],
                })
            run.log(log_data)

        if val_metrics is None:
            continue

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]

        if resolved_ckpt:
            _metric_map = {
                "val_accuracy": val_metrics["accuracy"],
                "val_ece":      val_metrics["ece"],
                "val_loss":     val_metrics["loss"],
            }
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints, top_k=top_k,
                checkpoint_path=resolved_ckpt, state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics":   val_metrics,
                    "config": runtime_cfg,
                },
                metric_name=checkpoint_metric.replace("val_", ""),
                metric_value=_metric_map.get(checkpoint_metric, val_metrics["accuracy"]),
                epoch=epoch, lower_is_better=lower_is_better,
            )

    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")

    # ── Test + OOD/CIFAR-C ────────────────────────────────────────────────────
    if saved_checkpoints:
        import shutil
        best_ckpt = torch.load(saved_checkpoints[0]["path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_model_path = saved_checkpoints[0]["path"].parent / "best_model.pt"
        shutil.copy2(saved_checkpoints[0]["path"], best_model_path)
        print(f"Best model saved to: {best_model_path}")

        print("\nEvaluating best checkpoint on held-out test set...")
        test_metrics = evaluate(model, test_loader, device, num_mc_samples)
        print(
            f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
            f"Test NLL: {test_metrics['nll']:.4f} | "
            f"Test ECE: {test_metrics['ece']:.4f}"
        )

        if run is not None:
            run.log({
                "test/accuracy": test_metrics["accuracy"],
                "test/nll":      test_metrics["nll"],
                "test/ece":      test_metrics["ece"],
                "test/loss":     test_metrics["loss"],
            })

        if not smoke_test and cfg.get("ood", {}).get("enabled", True):
            print("\nRunning OOD + CIFAR-C evaluation...")
            from src.training.post_training_eval import run_full_ood_eval
            wrapper = ModelWrapper(model=model, has_cov=True,
                                   num_mc_samples=num_mc_samples, model_type="supcon_sngp")
            id_logits, id_probs, _, _ = collect_logits_and_probs(
                wrapper, test_loader, device, num_mc_samples)
            run_full_ood_eval(
                model=model, has_cov=True, id_logits=id_logits, id_probs=id_probs,
                cfg=cfg, device=device, run=run,
                num_mc_samples=num_mc_samples, model_type="supcon_sngp",
            )

    if run is not None:
        if best_val_acc >= 0.0:
            run.log({"best/val_accuracy": best_val_acc})
        if saved_checkpoints:
            run.log({f"best/{checkpoint_metric}": saved_checkpoints[0]["metric"]})
            import wandb
            artifact = wandb.Artifact("cifar10_ms_sngp_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]),
                              name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 MS Loss + SNGP")
    parser.add_argument("--config",   required=True)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None, dest="run_name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
        if cfg.get("output", {}).get("checkpoint_path"):
            p = Path(cfg["output"]["checkpoint_path"])
            cfg["output"]["checkpoint_path"] = str(p.parent / f"seed{args.seed}" / p.name)
    if args.run_name:
        cfg.setdefault("wandb", {})["run_name"] = args.run_name

    main(cfg)
