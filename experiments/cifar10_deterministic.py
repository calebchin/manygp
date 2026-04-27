"""
CIFAR-10 deterministic WRN-28-10 training (Deep Ensemble member).

Trains a standard Wide ResNet-28-10 with a linear classification head and
CrossEntropy loss. Used as one member of a Deep Ensemble; run 5 times with
different seeds and then evaluate with cifar10_deep_ensemble_eval.py.

After training, runs OOD detection (SVHN, CIFAR-100) and CIFAR-10-C
corruption robustness evaluation in the same W&B run.

Usage:
    python experiments/cifar10_deterministic.py \
        --config configs/experiment_april4_deterministic.yaml \
        --seed 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import CifarResNetClassifier
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs
from src.utils.model_loader import ModelWrapper


def update_topk_checkpoints(
    saved_checkpoints: list[dict],
    top_k: int,
    checkpoint_path: str,
    state: dict,
    metric_value: float,
    epoch: int,
    lower_is_better: bool = False,
) -> None:
    if top_k <= 0:
        return

    checkpoint_target = Path(checkpoint_path)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = checkpoint_target.parent / (
        f"{checkpoint_target.stem}_epoch{epoch:03d}_acc{metric_value:.4f}{checkpoint_target.suffix or '.pt'}"
    )

    if len(saved_checkpoints) < top_k:
        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
    else:
        if lower_is_better:
            worst = max(saved_checkpoints, key=lambda x: (x["metric"], -x["epoch"]))
            if metric_value >= worst["metric"]:
                return
        else:
            worst = min(saved_checkpoints, key=lambda x: (x["metric"], -x["epoch"]))
            if metric_value <= worst["metric"]:
                return

        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
        worst["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst)

    saved_checkpoints.sort(key=lambda x: x["metric"], reverse=not lower_is_better)


def train_epoch(model, loader, optimizer, loss_fn, device, epoch, run=None, log_every_steps=None, global_step=0):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"Det Epoch {epoch}", leave=False)
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
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if run is not None and log_every_steps and global_step % log_every_steps == 0:
            run.log({"train/step_loss": loss.item(), "train/global_step": global_step})

    return running_loss / len(loader), total_correct / total_examples, global_step


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_nll = 0.0
    all_probs = []
    all_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        log_probs = probs.clamp_min(1e-12).log()

        running_loss += loss.item()
        total_correct += (probs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
        total_nll += -log_probs.gather(1, labels.unsqueeze(1)).sum().item()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs_t = torch.cat(all_probs)
    all_labels_t = torch.cat(all_labels)
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

    seed = cfg.get("training", {}).get("seed")
    if seed is not None:
        import random, numpy as np
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
            project=wandb_cfg.get("project", "deterministic"),
            entity=wandb_cfg.get("entity") or "sta414manygp",
            name=wandb_cfg.get("run_name") or None,
            config=cfg,
        )

    data_cfg = cfg["data"]
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_cifar10_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    model_cfg = cfg["model"]
    model = CifarResNetClassifier(
        widen_factor=model_cfg.get("widen_factor", 10),
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1 if smoke_test else train_cfg["epochs"]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps")
    output_cfg = cfg.get("output", {})
    checkpoint_path = output_cfg.get("checkpoint_path")
    top_k = output_cfg.get("top_k", 1)
    saved_checkpoints: list[dict] = []
    best_acc = -1.0
    global_step = 0

    epoch_progress = tqdm(range(1, num_epochs + 1), desc="Epoch", leave=True)
    for epoch in epoch_progress:
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            run=run, log_every_steps=log_every_steps, global_step=global_step,
        )
        scheduler.step()

        val_loss = val_acc = val_nll = val_ece = None
        if epoch % eval_interval == 0 or epoch == num_epochs:
            metrics = evaluate(model, val_loader, device)
            val_loss, val_acc, val_nll, val_ece = (
                metrics["loss"], metrics["accuracy"], metrics["nll"], metrics["ece"]
            )
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | "
                f"Val NLL: {val_nll:.4f} | Val ECE: {val_ece:.4f}"
            )
            epoch_progress.set_postfix(val_acc=f"{val_acc * 100:.2f}%")
        else:
            print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")

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

        if checkpoint_path:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc, "val_loss": val_loss,
                "val_nll": val_nll, "val_ece": val_ece,
                "config": cfg,
            }
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints, top_k=top_k,
                checkpoint_path=checkpoint_path, state=state,
                metric_value=val_acc, epoch=epoch,
            )

    if best_acc >= 0.0:
        print(f"Best validation accuracy: {best_acc * 100:.2f}%")

    # ── Test evaluation + OOD/CIFAR-C ────────────────────────────────────────
    if saved_checkpoints:
        import shutil
        best_ckpt = torch.load(saved_checkpoints[0]["path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_model_path = saved_checkpoints[0]["path"].parent / "best_model.pt"
        shutil.copy2(saved_checkpoints[0]["path"], best_model_path)
        print(f"Best model saved to: {best_model_path}")

        print("\nEvaluating best checkpoint on held-out test set...")
        test_metrics = evaluate(model, test_loader, device)
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
            wrapper = ModelWrapper(model=model, has_cov=False, model_type="classifier")
            id_logits, id_probs, _, _ = collect_logits_and_probs(wrapper, test_loader, device)
            run_full_ood_eval(
                model=model, has_cov=False, id_logits=id_logits, id_probs=id_probs,
                cfg=cfg, device=device, run=run, model_type="classifier",
            )

    if run is not None:
        if best_acc >= 0.0:
            run.log({"best/val_accuracy": best_acc})
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 deterministic WRN-28-10")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
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
