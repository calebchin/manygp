"""
CIFAR-10 supervised contrastive training with k-NN validation.

Usage:
    python experiments/cifar10_supcon.py --config configs/cifar10_supcon.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.resnet import SupConResNet
from src.training.contrastive import SupConLoss, evaluate_knn, train_supcon


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
    train_loader, memory_loader, val_loader, train_dataset, val_dataset = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(train_dataset)}, val size: {len(val_dataset)}")

    model_cfg = cfg["model"]
    model = SupConResNet(
        embedding_dim=model_cfg["embedding_dim"],
        projection_dim=model_cfg["projection_dim"],
        projection_hidden_dim=model_cfg["projection_hidden_dim"],
        width=model_cfg["width"],
    ).to(device)

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
    best_state = None
    num_epochs = 1 if smoke_test else train_cfg["epochs"]

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss = train_supcon(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            show_progress=True,
        )
        metrics = evaluate_knn(
            model=model,
            memory_loader=memory_loader,
            val_loader=val_loader,
            device=device,
            k=train_cfg["knn_k"],
            temperature=train_cfg["knn_temperature"],
        )
        knn_acc = metrics["knn_accuracy"]
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"SupCon Loss: {train_loss:.4f} | "
            f"k-NN Accuracy: {knn_acc * 100:.2f}%"
        )
        epoch_progress.set_postfix(loss=f"{train_loss:.4f}", knn=f"{knn_acc * 100:.2f}%")

        if run is not None:
            run.log({
                "train/loss": train_loss,
                "eval/knn_accuracy": knn_acc,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
            })

        if knn_acc > best_acc:
            best_acc = knn_acc
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "knn_accuracy": knn_acc,
                "config": cfg,
            }

    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    if checkpoint_path and best_state is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f"Saved best checkpoint to: {checkpoint_path}")

    print(f"Best k-NN accuracy: {best_acc * 100:.2f}%")
    if run is not None:
        run.log({"best/knn_accuracy": best_acc})
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 supervised contrastive experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
