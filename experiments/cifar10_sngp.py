"""
CIFAR-10 SNGP training with a spectrally normalized ResNet backbone.

Usage:
    python experiments/cifar10_sngp.py --config configs/cifar10_sngp.yaml
    python experiments/cifar10_sngp.py --config configs/cifar10_sngp.yaml --train-dataset supcon_two_view
"""

import argparse
import copy
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders, get_cifar10_two_view_classification_loaders
from src.models.sngp import SNGPResNetClassifier, laplace_predictive_probs
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper
from src.utils.model_summary import print_model_summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[float, float, int]:
    model.train()
    model.reset_precision_matrix()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"SNGP Epoch {epoch}", leave=False, disable=not show_progress)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if images.ndim == 5:
            batch_size, num_views, channels, height, width = images.shape
            images = images.view(batch_size * num_views, channels, height, width)
            labels = labels.repeat_interleave(num_views)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images, update_precision=True)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if run is not None and log_every_steps is not None and log_every_steps > 0:
            if global_step % log_every_steps == 0:
                run.log({
                    "train/step_loss": loss.item(),
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/lr_step": optimizer.param_groups[0]["lr"],
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
        "loss": running_loss / len(loader),
        "accuracy": total_correct / total_examples,
        "nll": total_nll / total_examples,
        "ece": ece,
    }


def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    seed = cfg["experiment"].get("seed")
    if seed is not None:
        set_seed(seed)
        print(f"Seed: {seed}")

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
    train_dataset_name = data_cfg.get("train_dataset")
    if train_dataset_name is None:
        train_dataset_name = "supcon_two_view" if data_cfg.get("use_supcon_augmentations", False) else "standard"

    if train_dataset_name == "supcon_two_view":
        train_loader, val_loader, train_dataset, val_dataset = get_cifar10_two_view_classification_loaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            smoke_test=smoke_test,
        )
    elif train_dataset_name == "standard":
        train_loader, val_loader, train_dataset, val_dataset = get_cifar10_loaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            smoke_test=smoke_test,
        )
    else:
        raise ValueError(
            f"Unsupported data.train_dataset={train_dataset_name!r}. "
            "Expected one of: 'standard', 'supcon_two_view'."
        )
    print(f"Train size: {len(train_dataset)}, val size: {len(val_dataset)}")
    print(f"Training dataset: {train_dataset_name}")

    model_cfg = cfg["model"]
    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        width=model_cfg["width"],
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
    best_nll = float("inf")
    best_ece = float("inf")
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    num_mc_samples = train_cfg.get("num_mc_samples", 10)
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    resolved_checkpoint_path = resolve_timestamped_checkpoint_path(checkpoint_path) if checkpoint_path else None
    top_k = output_cfg.get("top_k", 1)
    saved_checkpoints: list[dict] = []
    global_step = 0

    if resolved_checkpoint_path is not None:
        print(f"Checkpoint directory: {Path(resolved_checkpoint_path).parent}")

    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("output", {})["resolved_checkpoint_path"] = resolved_checkpoint_path

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss, train_acc, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            show_progress=True,
            run=run,
            log_every_steps=log_every_steps,
            global_step=global_step,
        )
        scheduler.step()

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        val_loss = None
        val_acc = None
        val_nll = None
        val_ece = None
        if should_evaluate:
            metrics = evaluate_sngp(
                model=model,
                loader=val_loader,
                device=device,
                num_mc_samples=num_mc_samples,
            )
            val_loss = metrics["loss"]
            val_acc = metrics["accuracy"]
            val_nll = metrics["nll"]
            val_ece = metrics["ece"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Val NLL: {val_nll:.4f} | "
                f"Val ECE: {val_ece:.4f}"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc * 100:.2f}%",
                val_acc=f"{val_acc * 100:.2f}%",
                val_ece=f"{val_ece:.4f}",
            )
        else:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}%"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc * 100:.2f}%",
            )

        if run is not None:
            log_data = {
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            if val_loss is not None and val_acc is not None and val_nll is not None and val_ece is not None:
                log_data["eval/loss"] = val_loss
                log_data["eval/accuracy"] = val_acc
                log_data["eval/nll"] = val_nll
                log_data["eval/ece"] = val_ece
            run.log(log_data)

        if val_acc is None:
            continue

        if val_acc > best_acc:
            best_acc = val_acc
        if val_nll is not None:
            best_nll = min(best_nll, val_nll)
        if val_ece is not None:
            best_ece = min(best_ece, val_ece)
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "val_nll": val_nll,
            "val_ece": val_ece,
            "config": runtime_cfg,
        }
        if resolved_checkpoint_path:
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints,
                top_k=top_k,
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                metric_name="acc",
                metric_value=val_acc,
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

    if best_acc >= 0.0:
        print(f"Best validation accuracy: {best_acc * 100:.2f}%")
    else:
        print("Best validation accuracy: not evaluated")

    if best_nll < float("inf"):
        print(f"Best validation NLL: {best_nll:.4f}")
    else:
        print("Best validation NLL: not evaluated")

    if best_ece < float("inf"):
        print(f"Best validation ECE: {best_ece:.4f}")
    else:
        print("Best validation ECE: not evaluated")

    test_metrics: dict[str, float] | None = None
    if saved_checkpoints:
        best_ckpt = torch.load(saved_checkpoints[0]["path"], map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        print("\nEvaluating best checkpoint on held-out evaluation split...")
        test_metrics = evaluate_sngp(model, val_loader, device, num_mc_samples)
        print(
            f"Test Acc: {test_metrics['accuracy'] * 100:.2f}% | "
            f"Test NLL: {test_metrics['nll']:.4f} | "
            f"Test ECE: {test_metrics['ece']:.4f}"
        )

        if run is not None:
            run.log({
                "test/accuracy": test_metrics["accuracy"],
                "test/nll": test_metrics["nll"],
                "test/ece": test_metrics["ece"],
                "test/loss": test_metrics["loss"],
            })

        if not smoke_test and cfg.get("ood", {}).get("enabled", True):
            print("\nRunning OOD evaluation (SVHN + CIFAR-100)...")
            from src.training.post_training_eval import run_ood_eval

            wrapper = ModelWrapper(
                model=model,
                has_cov=True,
                num_mc_samples=num_mc_samples,
                model_type="sngp",
            )
            id_logits, id_probs, _, _ = collect_logits_and_probs(
                wrapper,
                val_loader,
                device,
                num_mc_samples,
            )
            run_ood_eval(
                model=model,
                has_cov=True,
                id_logits=id_logits,
                id_probs=id_probs,
                cfg=cfg,
                device=device,
                run=run,
                num_mc_samples=num_mc_samples,
                model_type="sngp",
            )

    if run is not None:
        if best_acc >= 0.0:
            run.log({"best/val_accuracy": best_acc})
        if best_nll < float("inf"):
            run.log({"best/val_nll": best_nll})
        if best_ece < float("inf"):
            run.log({"best/val_ece": best_ece})
        if saved_checkpoints:
            import wandb

            artifact = wandb.Artifact("cifar10_sngp_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 SNGP experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--train-dataset",
        choices=("standard", "supcon_two_view"),
        default=None,
        help="Override data.train_dataset",
    )
    parser.add_argument(
        "--use-supcon-augmentations",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Backward-compatible override mapped to data.train_dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for experiment.seed",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.use_supcon_augmentations is not None:
        cfg.setdefault("data", {})["train_dataset"] = (
            "supcon_two_view" if args.use_supcon_augmentations else "standard"
        )
    if args.train_dataset is not None:
        cfg.setdefault("data", {})["train_dataset"] = args.train_dataset
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed

    main(cfg)
