"""
CIFAR-10 SNGP with SupCon-style strong data augmentation (CE loss only).

This is experiment 3: same SNGP architecture (spectrally normalized WRN-28-10 +
GP head) as experiment 1, but trained with the two-view strong augmentation
pipeline from SupCon (experiment 2). No contrastive loss is used — only CE.

Both augmented views of each image are fed through the model and CE loss is
computed over all views, effectively doubling the training signal with diverse
augmentations per step.

After training, runs OOD detection (SVHN, CIFAR-100) and CIFAR-10-C corruption
robustness evaluation in the same W&B run.

Usage:
    python experiments/cifar10_sngp_augmented.py \\
        --config configs/experiment_april2_sngp_augmented.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.sngp import SNGPResNetClassifier, laplace_predictive_probs
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper
from src.utils.model_summary import print_model_summary


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
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[float, float, int]:
    """Train one epoch with two-view augmented batches using CE loss only."""
    model.train()
    # NOTE: precision matrix reset is done once before the training loop starts.
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"SNGP-Aug Epoch {epoch}", leave=False, disable=not show_progress)
    for views, labels in progress:
        labels = labels.to(device, non_blocking=True)
        batch_size, num_views, channels, height, width = views.shape
        # Flatten both views into the batch dimension
        views = views.to(device, non_blocking=True).view(batch_size * num_views, channels, height, width)
        ce_labels = labels.repeat_interleave(num_views)

        optimizer.zero_grad(set_to_none=True)
        logits = model(views, update_precision=True)
        loss = loss_fn(logits, ce_labels)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == ce_labels).sum().item()
        total_examples += ce_labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if run is not None and log_every_steps is not None and log_every_steps > 0:
            if global_step % log_every_steps == 0:
                run.log({
                    "train/step_loss":   loss.item(),
                    "train/global_step": global_step,
                    "train/epoch":       epoch,
                    "train/lr_step":     optimizer.param_groups[0]["lr"],
                })

    return running_loss / len(loader), total_correct / total_examples, global_step


@torch.no_grad()
def evaluate_sngp(
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
    # Use the SupCon data pipeline for strong two-view augmentation
    train_loader, _, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    model_cfg = cfg["model"]
    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dim=model_cfg["hidden_dim"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
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
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    num_mc_samples = train_cfg.get("num_mc_samples", 10)
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    top_k = output_cfg.get("top_k", 1)
    checkpoint_metric = output_cfg.get("checkpoint_metric", "val_ece")
    lower_is_better = checkpoint_metric in ("val_ece", "val_loss")
    saved_checkpoints: list[dict] = []
    global_step = 0

    # Single precision-matrix reset before training begins (not per-epoch).
    model.reset_precision_matrix()

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss, train_acc, global_step = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            device=device, epoch=epoch, show_progress=True, run=run,
            log_every_steps=log_every_steps, global_step=global_step,
        )
        scheduler.step()

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        val_loss = val_acc = val_nll = val_ece = None
        if should_evaluate:
            metrics = evaluate_sngp(model=model, loader=val_loader, device=device, num_mc_samples=num_mc_samples)
            val_loss, val_acc, val_nll, val_ece = (
                metrics["loss"], metrics["accuracy"], metrics["nll"], metrics["ece"]
            )
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                f"Val NLL: {val_nll:.4f} | Val ECE: {val_ece:.4f}"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc * 100:.2f}%",
                val_acc=f"{val_acc * 100:.2f}%",
            )
        else:
            print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            epoch_progress.set_postfix(train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc * 100:.2f}%")

        if run is not None:
            log_data = {
                "train/loss": train_loss, "train/accuracy": train_acc,
                "train/lr": optimizer.param_groups[0]["lr"], "train/epoch": epoch,
            }
            if val_loss is not None:
                log_data.update({"val/loss": val_loss, "val/accuracy": val_acc,
                                 "val/nll": val_nll, "val/ece": val_ece})
            run.log(log_data)

        if val_acc is None:
            continue

        if val_acc > best_acc:
            best_acc = val_acc
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc, "val_loss": val_loss, "val_nll": val_nll, "val_ece": val_ece,
            "config": cfg,
        }
        if checkpoint_path:
            _metric_map = {"val_ece": val_ece, "val_loss": val_loss, "val_accuracy": val_acc}
            _ckpt_val = _metric_map.get(checkpoint_metric, val_ece)
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints, top_k=top_k,
                checkpoint_path=checkpoint_path, state=checkpoint_state,
                metric_name=checkpoint_metric.replace("val_", ""), metric_value=_ckpt_val, epoch=epoch,
                lower_is_better=lower_is_better,
            )

    if best_acc >= 0.0:
        print(f"Best validation accuracy: {best_acc * 100:.2f}%")

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
        test_metrics = evaluate_sngp(model, test_loader, device, num_mc_samples)
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
            wrapper = ModelWrapper(model=model, has_cov=True, num_mc_samples=num_mc_samples, model_type="sngp_augmented")
            id_logits, id_probs, _, _ = collect_logits_and_probs(wrapper, test_loader, device, num_mc_samples)
            run_full_ood_eval(
                model=model, has_cov=True, id_logits=id_logits, id_probs=id_probs,
                cfg=cfg, device=device, run=run, num_mc_samples=num_mc_samples, model_type="sngp_augmented",
            )

    if run is not None:
        if best_acc >= 0.0:
            run.log({"best/val_accuracy": best_acc})
        if saved_checkpoints:
            run.log({f"best/{checkpoint_metric}": saved_checkpoints[0]["metric"]})
            import wandb
            artifact = wandb.Artifact("cifar10_sngp_augmented_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 SNGP + SupCon augmentation (CE only)")
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
