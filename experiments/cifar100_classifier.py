"""
CIFAR-100 supervised classifier training with cross-entropy loss.

Usage:
    python experiments/cifar100_classifier.py --config configs/cifar100_classifier.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar100 import get_cifar100_loaders
from src.models.resnet import CifarResNetClassifier
from src.utils.model_summary import print_model_summary


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
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"CE Epoch {epoch}", leave=False, disable=not show_progress)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
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

    avg_loss = running_loss / len(loader)
    accuracy = total_correct / total_examples
    return avg_loss, accuracy, global_step


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)

        running_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return {
        "loss": running_loss / len(loader),
        "accuracy": total_correct / total_examples,
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
    train_loader, val_loader, train_dataset, val_dataset = get_cifar100_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(train_dataset)}, val size: {len(val_dataset)}")

    model_cfg = cfg["model"]
    model = CifarResNetClassifier(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        width=model_cfg["width"],
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
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    top_k = output_cfg.get("top_k", 1)
    saved_checkpoints: list[dict] = []
    global_step = 0

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
        if should_evaluate:
            metrics = evaluate_classifier(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            val_loss = metrics["loss"]
            val_acc = metrics["accuracy"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc * 100:.2f}%",
                val_acc=f"{val_acc * 100:.2f}%",
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
            if val_loss is not None and val_acc is not None:
                log_data["eval/loss"] = val_loss
                log_data["eval/accuracy"] = val_acc
            run.log(log_data)

        if val_acc is None:
            continue

        if val_acc > best_acc:
            best_acc = val_acc
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "config": cfg,
        }
        if checkpoint_path:
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints,
                top_k=top_k,
                checkpoint_path=checkpoint_path,
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

    if run is not None:
        if best_acc >= 0.0:
            run.log({"best/val_accuracy": best_acc})
        if saved_checkpoints:
            import wandb

            artifact = wandb.Artifact("cifar100_classifier_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-100 classifier baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
