"""
CIFAR-10 joint supervised contrastive + SNGP classification training.

After training, runs OOD detection (SVHN, CIFAR-100) and CIFAR-10-C corruption
robustness evaluation in the same W&B run.

Usage:
    python experiments/cifar10_supcon_sngp.py --config configs/cifar10_supcon_sngp.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
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
from src.training.contrastive import SupConLoss
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper
from src.utils.model_summary import print_model_summary


def resolve_timestamped_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_target = Path(checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(checkpoint_target.parent / timestamp / checkpoint_target.name)


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

    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = checkpoint_target.stem
    checkpoint_suffix = checkpoint_target.suffix or ".pt"
    candidate_path = checkpoint_target.parent / (
        f"{checkpoint_stem}_epoch{epoch:03d}_{metric_name}{metric_value:.4f}{checkpoint_suffix}"
    )

    if len(saved_checkpoints) < top_k:
        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
    else:
        if lower_is_better:
            worst_checkpoint = max(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
            if metric_value >= worst_checkpoint["metric"]:
                return
        else:
            worst_checkpoint = min(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
            if metric_value <= worst_checkpoint["metric"]:
                return

        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
        worst_checkpoint["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst_checkpoint)

    # For lower_is_better: sort ascending so [0] is the lowest (best) value.
    # For higher_is_better: sort descending so [0] is the highest (best) value.
    saved_checkpoints.sort(key=lambda item: item["metric"], reverse=not lower_is_better)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    supcon_loss_fn: torch.nn.Module,
    ce_loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    supcon_weight: float,
    ce_weight: float,
    show_progress: bool = True,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    model.train()
    # NOTE: precision matrix reset is done once before the training loop starts.
    running_total_loss = 0.0
    running_supcon_loss = 0.0
    running_ce_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"SupCon SNGP Epoch {epoch}", leave=False, disable=not show_progress)
    for views, labels in progress:
        labels = labels.to(device, non_blocking=True)
        batch_size, num_views, channels, height, width = views.shape
        views = views.to(device, non_blocking=True).view(batch_size * num_views, channels, height, width)
        ce_labels = labels.repeat_interleave(num_views)

        optimizer.zero_grad(set_to_none=True)
        logits, gp_inputs = model(views, update_precision=True, return_features=True)
        gp_inputs = gp_inputs.view(batch_size, num_views, -1)

        supcon_loss = supcon_loss_fn(gp_inputs, labels)
        ce_loss = ce_loss_fn(logits, ce_labels)
        total_loss = supcon_weight * supcon_loss + ce_weight * ce_loss
        total_loss.backward()
        optimizer.step()

        global_step += 1
        running_total_loss += total_loss.item()
        running_supcon_loss += supcon_loss.item()
        running_ce_loss += ce_loss.item()
        total_correct += (logits.argmax(dim=1) == ce_labels).sum().item()
        total_examples += ce_labels.size(0)
        progress.set_postfix(
            total=f"{total_loss.item():.4f}",
            supcon=f"{supcon_loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
        )

        if run is not None and log_every_steps is not None and log_every_steps > 0:
            if global_step % log_every_steps == 0:
                run.log({
                    "train/step_total_loss":  total_loss.item(),
                    "train/step_supcon_loss": supcon_loss.item(),
                    "train/step_ce_loss":     ce_loss.item(),
                    "train/global_step":      global_step,
                    "train/epoch":            epoch,
                    "train/lr_step":          optimizer.param_groups[0]["lr"],
                })

    return {
        "total_loss":  running_total_loss / len(loader),
        "supcon_loss": running_supcon_loss / len(loader),
        "ce_loss":     running_ce_loss / len(loader),
        "accuracy":    total_correct / total_examples,
    }, global_step


def mml_step(
    model: torch.nn.Module,
    gp_layer: torch.nn.Module,
    mml_loader_iter,
    mml_optimizer: torch.optim.Optimizer,
    device: torch.device,
    mml_steps: int = 1,
) -> float:
    """
    Run mml_steps gradient updates on log_length_scale using the Laplace MML objective.

    All model parameters except log_length_scale are frozen during this step.
    Returns the mean MML loss over the steps.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    gp_layer.log_length_scale.requires_grad_(True)

    total_loss = 0.0
    for _ in range(mml_steps):
        images, labels = next(mml_loader_iter)
        images, labels = images.to(device), labels.to(device)
        mml_optimizer.zero_grad()
        # Detach backbone features: gradient w.r.t. log_l comes through
        # random_weight / exp(log_l), not through the features themselves.
        with torch.no_grad():
            features = model.encode(images)
        loss = gp_layer.compute_laplace_log_mml(features, labels)
        loss.backward()
        mml_optimizer.step()
        total_loss += loss.item()

    for p in model.parameters():
        p.requires_grad_(True)
    return total_loss / mml_steps


def rebuild_precision_matrix(
    model: torch.nn.Module,
    train_loader,
    device: torch.device,
) -> None:
    """Reset and rebuild the precision matrix after a length scale update."""
    model.reset_precision_matrix()
    model.train()
    with torch.no_grad():
        for views, labels in train_loader:
            batch_size, num_views, channels, height, width = views.shape
            views = views.to(device).view(batch_size * num_views, channels, height, width)
            model(views, update_precision=True)


@torch.no_grad()
def evaluate_joint_sngp(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_mc_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_nll = 0.0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, variances = model(images, return_cov=True)
        probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        loss = -log_probs.gather(1, labels.unsqueeze(1)).mean()

        running_loss += loss.item()
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll += -log_probs.gather(1, labels.unsqueeze(1)).sum().item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t = torch.cat(all_probs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    ece = _classification_ece(all_probs_t, all_labels_t)

    return {
        "loss":     running_loss / len(loader),
        "accuracy": total_correct / total_examples,
        "nll":      total_nll / total_examples,
        "ece":      ece,
    }


def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = cfg.get("training", {}).get("seed", None)
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
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
    train_loader, _, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_cifar10_supcon_loaders(
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
        optimize_length_scale=model_cfg.get("optimize_length_scale", False),
    ).to(device)
    print_model_summary(model)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1 if smoke_test else train_cfg["epochs"],
    )
    supcon_loss_fn = SupConLoss(temperature=train_cfg["supcon_temperature"])
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    num_mc_samples = train_cfg.get("num_mc_samples", 10)
    supcon_weight = train_cfg.get("supcon_loss_weight", 1.0)
    ce_weight = train_cfg.get("ce_loss_weight", 1.0)
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    resolved_checkpoint_path = resolve_timestamped_checkpoint_path(checkpoint_path) if checkpoint_path else None
    top_k = output_cfg.get("top_k", 5)
    checkpoint_metric = output_cfg.get("checkpoint_metric", "val_ece")
    lower_is_better = checkpoint_metric in ("val_ece", "val_loss")
    saved_checkpoints: list[dict] = []
    global_step = 0

    # MML optimizer setup (only active when optimize_length_scale=True in model config)
    mml_cfg = cfg.get("mml", {})
    mml_enabled = (
        mml_cfg.get("enabled", False)
        and model_cfg.get("optimize_length_scale", False)
        and not smoke_test
    )
    mml_optimizer = None
    mml_loader_iter = None
    mml_interval = mml_cfg.get("interval_epochs", 1)
    mml_steps_per_interval = mml_cfg.get("steps_per_interval", 5)
    if mml_enabled:
        mml_optimizer = torch.optim.Adam(
            [model.classifier.log_length_scale],
            lr=mml_cfg.get("lr", 1e-3),
        )
        mml_loader_iter = iter(itertools.cycle(val_loader))

    # Single precision-matrix reset before training begins (not per-epoch).
    model.reset_precision_matrix()

    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_checkpoint_path

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_metrics, global_step = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            supcon_loss_fn=supcon_loss_fn, ce_loss_fn=ce_loss_fn, device=device,
            epoch=epoch, supcon_weight=supcon_weight, ce_weight=ce_weight,
            show_progress=True, run=run, log_every_steps=log_every_steps, global_step=global_step,
        )
        scheduler.step()

        if mml_enabled and epoch % mml_interval == 0:
            mml_loss = mml_step(
                model=model,
                gp_layer=model.classifier,
                mml_loader_iter=mml_loader_iter,
                mml_optimizer=mml_optimizer,
                device=device,
                mml_steps=mml_steps_per_interval,
            )
            current_ls = model.classifier.log_length_scale.exp().item()
            # Rebuild precision matrix under the updated length scale
            rebuild_precision_matrix(model, train_loader, device)
            if run is not None:
                run.log({"mml/loss": mml_loss, "mml/length_scale": current_ls, "train/epoch": epoch})

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        val_metrics = None
        if should_evaluate:
            val_metrics = evaluate_joint_sngp(model=model, loader=val_loader, device=device, num_mc_samples=num_mc_samples)
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Total: {train_metrics['total_loss']:.4f} | "
                f"Train SupCon: {train_metrics['supcon_loss']:.4f} | "
                f"Train CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
                f"Val NLL: {val_metrics['nll']:.4f}"
            )
            epoch_progress.set_postfix(
                train_total=f"{train_metrics['total_loss']:.4f}",
                train_acc=f"{train_metrics['accuracy'] * 100:.2f}%",
                val_acc=f"{val_metrics['accuracy'] * 100:.2f}%",
            )
        else:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Total: {train_metrics['total_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy'] * 100:.2f}%"
            )
            epoch_progress.set_postfix(
                train_total=f"{train_metrics['total_loss']:.4f}",
                train_acc=f"{train_metrics['accuracy'] * 100:.2f}%",
            )

        if run is not None:
            log_data = {
                "train/total_loss":  train_metrics["total_loss"],
                "train/supcon_loss": train_metrics["supcon_loss"],
                "train/ce_loss":     train_metrics["ce_loss"],
                "train/accuracy":    train_metrics["accuracy"],
                "train/lr":          optimizer.param_groups[0]["lr"],
                "train/epoch":       epoch,
            }
            if val_metrics is not None:
                log_data.update({"val/loss": val_metrics["loss"], "val/accuracy": val_metrics["accuracy"],
                                 "val/nll": val_metrics["nll"], "val/ece": val_metrics["ece"]})
            run.log(log_data)

        if val_metrics is None:
            continue

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": runtime_cfg,
            "log_length_scale": model.classifier.log_length_scale.item(),
        }
        if resolved_checkpoint_path:
            _metric_map = {"val_ece": val_metrics["ece"], "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]}
            _ckpt_val = _metric_map.get(checkpoint_metric, val_metrics["ece"])
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints, top_k=top_k,
                checkpoint_path=resolved_checkpoint_path, state=checkpoint_state,
                metric_name=checkpoint_metric.replace("val_", ""), metric_value=_ckpt_val, epoch=epoch,
                lower_is_better=lower_is_better,
            )

    if best_val_acc >= 0.0:
        print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")

    # ── Test evaluation + OOD/CIFAR-C (same W&B run) ─────────────────────────
    test_metrics: dict[str, float] | None = None
    if saved_checkpoints:
        import shutil
        best_ckpt = torch.load(saved_checkpoints[0]["path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_model_path = saved_checkpoints[0]["path"].parent / "best_model.pt"
        shutil.copy2(saved_checkpoints[0]["path"], best_model_path)
        print(f"Best model saved to: {best_model_path}")

        print("\nEvaluating best checkpoint on held-out test set...")
        test_metrics = evaluate_joint_sngp(model, test_loader, device, num_mc_samples)
        print(
            f"Test Acc: {test_metrics['accuracy'] * 100:.2f}% | "
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
            wrapper = ModelWrapper(model=model, has_cov=True, num_mc_samples=num_mc_samples, model_type="supcon_sngp")
            id_logits, id_probs, _, _ = collect_logits_and_probs(wrapper, test_loader, device, num_mc_samples)
            run_full_ood_eval(
                model=model, has_cov=True, id_logits=id_logits, id_probs=id_probs,
                cfg=cfg, device=device, run=run, num_mc_samples=num_mc_samples, model_type="supcon_sngp",
            )

    if run is not None:
        if best_val_acc >= 0.0:
            run.log({"best/val_accuracy": best_val_acc})
        if saved_checkpoints:
            run.log({f"best/{checkpoint_metric}": saved_checkpoints[0]["metric"]})
            import wandb
            artifact = wandb.Artifact("cifar10_supcon_sngp_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 joint SupCon + SNGP training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name",
                        help="W&B run name (overrides config)")
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
