"""
CIFAR-10 SimCLR self-supervised pretraining.

Pretrain a WideResNet-28-10 backbone with the SimCLR objective (NT-Xent loss).
The projection head is discarded after pretraining; only the backbone weights
are saved and can be loaded with ``load_simclr_backbone()`` for frozen-backbone
SNGP finetuning.

Usage:
    python experiments/cifar10_simclr_pretrain.py --config configs/cifar10_simclr_pretrain.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_simclr_loaders
from src.models.simclr import SimCLRModel
from src.training.contrastive import NTXentLoss, evaluate_knn
from src.utils.model_summary import print_model_summary


def train_epoch(
    model: SimCLRModel,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NTXentLoss,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc=f"SimCLR Epoch {epoch}", leave=False, disable=not show_progress)
    for views, _labels in progress:
        batch_size, num_views, C, H, W = views.shape
        views = views.to(device, non_blocking=True)
        # views[:, 0] = all view-1 images, views[:, 1] = all view-2 images.
        # permute to (num_views, B, C, H, W) then reshape to (2B, C, H, W) so
        # the first B rows are view-1 and the last B rows are view-2 — matching
        # what NTXentLoss expects when we chunk(2).
        flat_views = views.permute(1, 0, 2, 3, 4).reshape(batch_size * num_views, C, H, W)

        optimizer.zero_grad(set_to_none=True)
        projections = model(flat_views)
        z1, z2 = projections.chunk(2, dim=0)
        loss = loss_fn(z1, z2)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if run is not None and log_every_steps is not None and log_every_steps > 0:
            if global_step % log_every_steps == 0:
                run.log({
                    "train/step_loss": loss.item(),
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/lr_step": optimizer.param_groups[0]["lr"],
                })

    return running_loss / len(loader), global_step


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
    train_loader, memory_loader, train_dataset, memory_dataset = get_cifar10_simclr_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(train_dataset)}")

    model_cfg = cfg["model"]
    model = SimCLRModel(
        widen_factor=model_cfg["widen_factor"],
        embedding_dim=model_cfg["embedding_dim"],
        proj_hidden_dim=model_cfg["proj_hidden_dim"],
        proj_out_dim=model_cfg["proj_out_dim"],
    ).to(device)
    print_model_summary(model)

    train_cfg = cfg["training"]
    temperature = train_cfg.get("temperature", 0.5)
    loss_fn = NTXentLoss(temperature=temperature)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    log_every_steps = train_cfg.get("log_every_steps", None)
    knn_eval_interval = train_cfg.get("knn_eval_interval", 10)

    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")

    best_loss = float("inf")
    best_ckpt_path: Path | None = None

    global_step = 0
    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss, global_step = train_epoch(
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

        log_data: dict = {
            "train/loss": train_loss,
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch,
        }

        should_eval_knn = (epoch % knn_eval_interval == 0) or (epoch == num_epochs)
        knn_acc = None
        if should_eval_knn and not smoke_test:
            knn_metrics = evaluate_knn(
                model=model,
                memory_loader=memory_loader,
                val_loader=memory_loader,
                device=device,
            )
            knn_acc = knn_metrics["knn_accuracy"]
            log_data["eval/knn_accuracy"] = knn_acc

        if knn_acc is not None:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"kNN Acc: {knn_acc * 100:.2f}%"
            )
            epoch_progress.set_postfix(loss=f"{train_loss:.4f}", knn=f"{knn_acc * 100:.2f}%")
        else:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {train_loss:.4f}")
            epoch_progress.set_postfix(loss=f"{train_loss:.4f}")

        if run is not None:
            run.log(log_data)

        if checkpoint_path and epoch % 50 == 0 and epoch != num_epochs:
            ckpt_dir = Path(checkpoint_path).parent
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            new_ckpt_path = ckpt_dir / f"cifar10_simclr_epoch{epoch}.pt"
            torch.save(
                {
                    "encoder_state_dict": model.encoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "config": cfg,
                },
                new_ckpt_path,
            )
            print(f"Checkpoint saved at epoch {epoch} (loss={train_loss:.4f})")
            if train_loss < best_loss:
                if best_ckpt_path is not None and best_ckpt_path.exists():
                    best_ckpt_path.unlink()
                best_loss = train_loss
                best_ckpt_path = new_ckpt_path
            else:
                new_ckpt_path.unlink()

    # Save final checkpoint and upload best + final to wandb
    if checkpoint_path:
        out_path = Path(checkpoint_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state_dict": model.encoder.state_dict(),
                "epoch": num_epochs,
                "train_loss": train_loss,
                "config": cfg,
            },
            out_path,
        )
        print(f"Backbone checkpoint saved to {out_path}")

        if run is not None:
            import wandb

            artifact = wandb.Artifact("cifar10_simclr_backbone", type="model")
            artifact.add_file(str(out_path), name=out_path.name)
            if best_ckpt_path is not None and best_ckpt_path.exists():
                artifact.add_file(str(best_ckpt_path), name=best_ckpt_path.name)
            run.log_artifact(artifact)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 SimCLR self-supervised pretraining")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
