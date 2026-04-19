"""
CIFAR-10 joint supervised contrastive + SNGP classification training.

Usage:
    python experiments/cifar10_supcon_sngp.py --config configs/cifar10_supcon_sngp.yaml
    python experiments/cifar10_supcon_sngp.py --config configs/cifar10_supcon_sngp.yaml --supcon-loss-weight 0.5
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.sngp import laplace_predictive_probs
from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier
from src.training.contrastive import SupConLoss
from src.utils.model_summary import print_model_summary


def resolve_timestamped_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_target = Path(checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid4().hex[:8]
    return str(checkpoint_target.parent / f"{timestamp}_{random_suffix}" / checkpoint_target.name)


def update_topk_checkpoints(
    saved_checkpoints: list[dict],
    top_k: int,
    checkpoint_path: str,
    state: dict,
    metric_name: str,
    metric_value: float,
    epoch: int,
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
        worst_checkpoint = min(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
        if metric_value <= worst_checkpoint["metric"]:
            return

        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
        worst_checkpoint["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst_checkpoint)

    saved_checkpoints.sort(key=lambda item: item["metric"], reverse=True)


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
    model.reset_precision_matrix()

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
                    "train/step_total_loss": total_loss.item(),
                    "train/step_supcon_loss": supcon_loss.item(),
                    "train/step_ce_loss": ce_loss.item(),
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/lr_step": optimizer.param_groups[0]["lr"],
                })

    metrics = {
        "total_loss": running_total_loss / len(loader),
        "supcon_loss": running_supcon_loss / len(loader),
        "ce_loss": running_ce_loss / len(loader),
        "accuracy": total_correct / total_examples,
    }
    return metrics, global_step


@torch.no_grad()
def evaluate_joint_sngp(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_mc_samples: int = 10,
) -> dict[str, float]:
    model.eval()
    total_correct = 0
    total_examples = 0
    total_ce_loss = 0.0
    total_nll = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, variances = model(images, return_cov=True)
        probs = laplace_predictive_probs(logits, variances, num_mc_samples=num_mc_samples)
        log_probs = probs.clamp_min(1e-12).log()
        total_ce_loss += F.cross_entropy(logits, labels, reduction="sum").item()

        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll += -log_probs.gather(1, labels.unsqueeze(1)).sum().item()

    return {
        "loss": total_nll / total_examples,
        "accuracy": total_correct / total_examples,
        "ce_loss": total_ce_loss / total_examples,
        "nll": total_nll / total_examples,
    }


def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    train_loader, _, val_loader, train_dataset, val_dataset = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(train_dataset)}, val size: {len(val_dataset)}")

    model_cfg = cfg["model"]
    model = CifarResNetSupConSNGPClassifier(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        width=model_cfg["width"],
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
    optimizer = torch.optim.AdamW(
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
    best_val_nll = float("inf")
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
    saved_checkpoints: list[dict] = []
    global_step = 0

    if resolved_checkpoint_path is not None:
        print(f"Checkpoint directory: {Path(resolved_checkpoint_path).parent}")

    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_checkpoint_path

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_metrics, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            supcon_loss_fn=supcon_loss_fn,
            ce_loss_fn=ce_loss_fn,
            device=device,
            epoch=epoch,
            supcon_weight=supcon_weight,
            ce_weight=ce_weight,
            show_progress=True,
            run=run,
            log_every_steps=log_every_steps,
            global_step=global_step,
        )
        scheduler.step()

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        val_metrics = None
        if should_evaluate:
            val_metrics = evaluate_joint_sngp(
                model=model,
                loader=val_loader,
                device=device,
                num_mc_samples=num_mc_samples,
            )
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Total: {train_metrics['total_loss']:.4f} | "
                f"Train SupCon: {train_metrics['supcon_loss']:.4f} | "
                f"Train CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy'] * 100:.2f}% | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
                f"Val CE: {val_metrics['ce_loss']:.4f} | "
                f"Val NLL: {val_metrics['nll']:.4f}"
            )
            epoch_progress.set_postfix(
                train_total=f"{train_metrics['total_loss']:.4f}",
                train_acc=f"{train_metrics['accuracy'] * 100:.2f}%",
                val_acc=f"{val_metrics['accuracy'] * 100:.2f}%",
                val_nll=f"{val_metrics['nll']:.4f}",
            )
        else:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Total: {train_metrics['total_loss']:.4f} | "
                f"Train SupCon: {train_metrics['supcon_loss']:.4f} | "
                f"Train CE: {train_metrics['ce_loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy'] * 100:.2f}%"
            )
            epoch_progress.set_postfix(
                train_total=f"{train_metrics['total_loss']:.4f}",
                train_acc=f"{train_metrics['accuracy'] * 100:.2f}%",
            )

        if run is not None:
            log_data = {
                "train/total_loss": train_metrics["total_loss"],
                "train/supcon_loss": train_metrics["supcon_loss"],
                "train/ce_loss": train_metrics["ce_loss"],
                "train/accuracy": train_metrics["accuracy"],
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            if val_metrics is not None:
                log_data["val/loss"] = val_metrics["loss"]
                log_data["val/accuracy"] = val_metrics["accuracy"]
                log_data["val/ce_loss"] = val_metrics["ce_loss"]
                log_data["val/nll"] = val_metrics["nll"]
            run.log(log_data)

        if val_metrics is None:
            continue

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
        best_val_nll = min(best_val_nll, val_metrics["nll"])
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": runtime_cfg,
        }
        if resolved_checkpoint_path:
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints,
                top_k=top_k,
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                metric_name="acc",
                metric_value=val_metrics["accuracy"],
                epoch=epoch,
            )

    if saved_checkpoints:
        print("Saved top checkpoints:")
        for checkpoint in saved_checkpoints:
            print(
                f"  epoch {checkpoint['epoch']:3d} | "
                f"val acc {checkpoint['metric'] * 100:.2f}% | "
                f"{checkpoint['path']}"
            )

    if best_val_acc >= 0.0:
        print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    else:
        print("Best validation accuracy: not evaluated")

    if best_val_nll < float("inf"):
        print(f"Best validation NLL: {best_val_nll:.4f}")
    else:
        print("Best validation NLL: not evaluated")

    if run is not None:
        if best_val_acc >= 0.0:
            run.log({"best/val_accuracy": best_val_acc})
        if best_val_nll < float("inf"):
            run.log({"best/val_nll": best_val_nll})
        if saved_checkpoints:
            import wandb

            artifact = wandb.Artifact("cifar10_supcon_sngp_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 joint SupCon + SNGP training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--supcon-loss-weight",
        type=float,
        default=None,
        help="Optional override for training.supcon_loss_weight",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.supcon_loss_weight is not None:
        cfg.setdefault("training", {})["supcon_loss_weight"] = args.supcon_loss_weight

    main(cfg)
