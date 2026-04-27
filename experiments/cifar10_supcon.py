"""
CIFAR-10 supervised contrastive training with k-NN validation.

Usage:
    python experiments/cifar10_supcon.py --config configs/cifar10_supcon.yaml
"""

import argparse
import os
import random
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.resnet import SupConResNet
from src.training.contrastive import SupConLoss, evaluate_knn, evaluate_supcon_loss, train_supcon
from src.utils.model_summary import print_model_summary


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_randomized_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_target = Path(checkpoint_path)
    random_suffix = uuid4().hex[:8]
    checkpoint_dir = checkpoint_target.parent
    resolved_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_{random_suffix}"
    return str(resolved_dir / checkpoint_target.name)


def update_topk_checkpoints(
    saved_checkpoints: list[dict],
    top_k: int,
    checkpoint_path: str,
    state: dict,
    metric_name: str,
    metric_value: float,
    epoch: int,
    higher_is_better: bool = True,
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
        if higher_is_better:
            worst_checkpoint = min(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
            should_save = metric_value > worst_checkpoint["metric"]
            reverse = True
        else:
            worst_checkpoint = max(saved_checkpoints, key=lambda item: (item["metric"], item["epoch"]))
            should_save = metric_value < worst_checkpoint["metric"]
            reverse = False

        if not should_save:
            return

        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
        worst_checkpoint["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst_checkpoint)
        saved_checkpoints.sort(key=lambda item: item["metric"], reverse=reverse)
        return

    saved_checkpoints.sort(key=lambda item: item["metric"], reverse=higher_is_better)


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
    train_loader, memory_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(
        f"Train size: {len(train_dataset)}, "
        f"val size: {len(val_dataset)}, "
        f"test size: {len(test_dataset)}"
    )

    model_cfg = cfg["model"]
    model = SupConResNet(
        embedding_dim=model_cfg["embedding_dim"],
        projection_dim=model_cfg["projection_dim"],
        projection_hidden_dim=model_cfg["projection_hidden_dim"],
        width=model_cfg["width"],
        spec_norm_bound=model_cfg.get("spec_norm_bound", 0.95),
        use_projection_head=model_cfg.get("use_projection_head", True),
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
    loss_fn = SupConLoss(temperature=train_cfg["supcon_temperature"])

    best_acc = -1.0
    best_val_supcon_loss = float("inf")
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    resolved_checkpoint_path = resolve_randomized_checkpoint_path(checkpoint_path) if checkpoint_path else None
    top_k = output_cfg.get("top_k", 1)
    saved_checkpoints: list[dict] = []
    global_step = 0

    if resolved_checkpoint_path is not None:
        print(f"Checkpoint directory: {Path(resolved_checkpoint_path).parent}")

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss, global_step = train_supcon(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            show_progress=True,
            run=run,
            log_every_steps=log_every_steps,
            global_step=global_step,
        )

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        knn_acc = None
        val_supcon_loss = None
        if should_evaluate:
            knn_metrics = evaluate_knn(
                model=model,
                memory_loader=memory_loader,
                val_loader=val_loader,
                device=device,
                k=train_cfg["knn_k"],
                temperature=train_cfg["knn_temperature"],
            )
            supcon_metrics = evaluate_supcon_loss(
                model=model,
                loader=val_supcon_loader,
                loss_fn=loss_fn,
                device=device,
            )
            knn_acc = knn_metrics["knn_accuracy"]
            val_supcon_loss = supcon_metrics["supcon_loss"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"SupCon Loss: {train_loss:.4f} | "
                f"Val SupCon Loss: {val_supcon_loss:.4f} | "
                f"k-NN Accuracy: {knn_acc * 100:.2f}%"
            )
            epoch_progress.set_postfix(
                loss=f"{train_loss:.4f}",
                val_loss=f"{val_supcon_loss:.4f}",
                knn=f"{knn_acc * 100:.2f}%",
            )
        else:
            print(f"Epoch {epoch:3d}/{num_epochs} | SupCon Loss: {train_loss:.4f}")
            epoch_progress.set_postfix(loss=f"{train_loss:.4f}")

        if run is not None:
            log_data = {
                "train/loss": train_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            }
            if knn_acc is not None:
                log_data["eval/knn_accuracy"] = knn_acc
            if val_supcon_loss is not None:
                log_data["eval/supcon_loss"] = val_supcon_loss
            run.log(log_data)

        if knn_acc is None:
            continue

        if knn_acc > best_acc:
            best_acc = knn_acc
        if val_supcon_loss is not None and val_supcon_loss < best_val_supcon_loss:
            best_val_supcon_loss = val_supcon_loss
        checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "knn_accuracy": knn_acc,
                "val_supcon_loss": val_supcon_loss,
                "resolved_checkpoint_path": resolved_checkpoint_path,
                "config": cfg,
            }
        if resolved_checkpoint_path:
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints,
                top_k=top_k,
                checkpoint_path=resolved_checkpoint_path,
                state=checkpoint_state,
                metric_name="knn",
                metric_value=knn_acc,
                epoch=epoch,
            )

    if saved_checkpoints:
        print("Saved top checkpoints:")
        for checkpoint in saved_checkpoints:
            print(f"  epoch {checkpoint['epoch']:3d} | k-NN acc {checkpoint['metric'] * 100:.2f}% | {checkpoint['path']}")

    if best_acc >= 0.0:
        print(f"Best k-NN accuracy: {best_acc * 100:.2f}%")
    else:
        print("Best k-NN accuracy: not evaluated")
    if best_val_supcon_loss < float("inf"):
        print(f"Best val SupCon loss: {best_val_supcon_loss:.4f}")
    else:
        print("Best val SupCon loss: not evaluated")
    if run is not None:
        if best_acc >= 0.0:
            run.log({"best/knn_accuracy": best_acc})
        if best_val_supcon_loss < float("inf"):
            run.log({"best/supcon_loss": best_val_supcon_loss})
        if saved_checkpoints:
            import wandb

            artifact = wandb.Artifact("cifar10_supcon_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 supervised contrastive experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for experiment.seed",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional override for wandb.run_name",
    )
    projection_group = parser.add_mutually_exclusive_group()
    projection_group.add_argument(
        "--use-projection-head",
        dest="use_projection_head",
        action="store_true",
        help="Override model.use_projection_head to true",
    )
    projection_group.add_argument(
        "--no-use-projection-head",
        dest="use_projection_head",
        action="store_false",
        help="Override model.use_projection_head to false",
    )
    parser.set_defaults(use_projection_head=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.run_name is not None:
        cfg.setdefault("wandb", {})["run_name"] = args.run_name
    if args.use_projection_head is not None:
        cfg.setdefault("model", {})["use_projection_head"] = args.use_projection_head

    main(cfg)
