"""
CIFAR-10 Deep Ensemble baseline.

Trains N independent CifarResNetClassifier models with different random seeds,
then aggregates their softmax outputs for ensemble predictions.  After training,
evaluates the ensemble on the held-out test set and runs inline OOD detection
against SVHN and CIFAR-100.

Usage:
    python experiments/cifar10_deep_ensemble.py --config configs/cifar10_deep_ensemble.yaml
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import CifarResNetClassifier
from src.training.evaluate import _classification_ece
from src.training.ood_evaluate import collect_logits_and_probs, evaluate_ood_split
from src.utils.model_summary import print_model_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    run=None,
    log_prefix: str = "train",
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
                    f"{log_prefix}/step_loss": loss.item(),
                    f"{log_prefix}/global_step": global_step,
                    f"{log_prefix}/epoch": epoch,
                })

    return running_loss / len(loader), total_correct / total_examples, global_step


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


@torch.no_grad()
def get_softmax_probs(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (probs, labels) for an entire loader."""
    model.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def train_single_member(
    seed: int,
    member_idx: int,
    train_loader,
    val_loader,
    model_cfg: dict,
    train_cfg: dict,
    output_cfg: dict,
    device: torch.device,
    smoke_test: bool,
    run=None,
    log_every_steps: int | None = None,
) -> tuple[torch.nn.Module, float, Path | None]:
    """Train one ensemble member.  Returns (model, best_val_acc, best_ckpt_path)."""
    set_seed(seed)

    model = CifarResNetClassifier(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        width=model_cfg["width"],
    ).to(device)

    if member_idx == 0:
        print_model_summary(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 5)
    checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "./checkpoints_ensemble_cifar10/"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path: Path | None = None
    best_val_acc = -1.0
    global_step = 0

    epoch_progress = tqdm(range(1, num_epochs + 1), desc=f"Member {member_idx}", leave=True)
    for epoch in epoch_progress:
        log_prefix = f"member_{member_idx}/train"
        train_loss, train_acc, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            show_progress=False,
            run=run,
            log_prefix=log_prefix,
            log_every_steps=log_every_steps,
            global_step=global_step,
        )
        scheduler.step()

        should_evaluate = epoch % eval_interval == 0 or epoch == num_epochs
        if should_evaluate:
            val_metrics = evaluate_classifier(model, val_loader, loss_fn, device)
            val_acc = val_metrics["accuracy"]

            print(
                f"  [M{member_idx}] Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}%"
            )
            epoch_progress.set_postfix(val_acc=f"{val_acc * 100:.2f}%")

            if run is not None:
                run.log({
                    f"member_{member_idx}/train/loss": train_loss,
                    f"member_{member_idx}/train/accuracy": train_acc,
                    f"member_{member_idx}/val/accuracy": val_acc,
                    f"member_{member_idx}/val/loss": val_metrics["loss"],
                    f"member_{member_idx}/train/epoch": epoch,
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_ckpt_path = checkpoint_dir / f"member_{member_idx}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "seed": seed,
                    "member_idx": member_idx,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                }, best_ckpt_path)

    print(f"  [M{member_idx}] Best val acc: {best_val_acc * 100:.2f}% — {best_ckpt_path}")
    return model, best_val_acc, best_ckpt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict) -> None:
    smoke_test = cfg["experiment"]["smoke_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        import wandb
        run = wandb.init(
            project=wandb_cfg.get("project", "Updated_run"),
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
    print(f"Train size: {len(train_dataset)}, val size: {len(val_dataset)}, test size: {len(test_dataset)}")

    ensemble_cfg = cfg.get("ensemble", {})
    seeds = ensemble_cfg.get("seeds", [0, 1, 2, 3, 4])
    num_members = ensemble_cfg.get("num_members", len(seeds))
    seeds = seeds[:num_members]

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg.get("output", {})
    log_every_steps = train_cfg.get("log_every_steps", None)

    # ── Train all members ────────────────────────────────────────────────────
    trained_models: list[torch.nn.Module] = []
    member_val_accs: list[float] = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Training member {i} / seed {seed}")
        print(f"{'='*60}")
        model, best_val_acc, _ = train_single_member(
            seed=seed,
            member_idx=i,
            train_loader=train_loader,
            val_loader=val_loader,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            output_cfg=output_cfg,
            device=device,
            smoke_test=smoke_test,
            run=run,
            log_every_steps=log_every_steps,
        )
        trained_models.append(model)
        member_val_accs.append(best_val_acc)

    # ── Ensemble test evaluation ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("Ensemble evaluation on held-out test set")
    print("="*60)

    all_member_probs: list[torch.Tensor] = []
    test_labels: torch.Tensor | None = None

    for i, model in enumerate(trained_models):
        probs, labels = get_softmax_probs(model, test_loader, device)
        all_member_probs.append(probs)
        if test_labels is None:
            test_labels = labels
        member_acc = (probs.argmax(1) == labels).float().mean().item()
        print(f"  Member {i} test accuracy: {member_acc * 100:.2f}%")
        if run is not None:
            run.log({f"member_{i}/test/accuracy": member_acc})

    assert test_labels is not None
    ens_probs = torch.stack(all_member_probs).mean(0)       # (N, C)
    ens_acc = (ens_probs.argmax(1) == test_labels).float().mean().item()
    ens_nll = -ens_probs.clamp_min(1e-12).log()[
        torch.arange(len(test_labels)), test_labels
    ].mean().item()
    ens_ece = _classification_ece(ens_probs, test_labels)

    print(f"\nEnsemble test accuracy : {ens_acc * 100:.2f}%")
    print(f"Ensemble test NLL      : {ens_nll:.4f}")
    print(f"Ensemble test ECE      : {ens_ece:.4f}")

    # ── Inline OOD evaluation ────────────────────────────────────────────────
    # Build a thin wrapper so collect_logits_and_probs can call the ensemble
    class EnsembleWrapper:
        model_type = "ensemble"
        has_cov = False

        def __init__(self, models, dev):
            self._models = models
            self._dev = dev

        def __call__(self, images):
            probs_list = []
            for m in self._models:
                m.eval()
                with torch.no_grad():
                    p = torch.softmax(m(images.to(self._dev)), dim=1)
                probs_list.append(p)
            avg = torch.stack(probs_list).mean(0)
            # Return logits-like tensor (log of averaged probs) and avg probs
            return avg.log(), avg

        def eval(self):
            for m in self._models:
                m.eval()
            return self

    ens_wrapper = EnsembleWrapper(trained_models, device)

    ood_results: dict[str, dict] = {}

    # Collect ID probs from the ensemble on the test set (already computed above)
    id_probs = ens_probs          # (N, C)
    id_logits = id_probs.log()    # used only for DS score

    try:
        from src.data.svhn import get_svhn_loader
        print("\nRunning OOD detection against SVHN...")
        svhn_loader = get_svhn_loader(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 0),
            id_normalization="cifar10",
        )
        svhn_probs_list = [get_softmax_probs(m, svhn_loader, device)[0] for m in trained_models]
        svhn_probs = torch.stack(svhn_probs_list).mean(0)
        svhn_logits = svhn_probs.log()
        svhn_metrics = evaluate_ood_split(id_logits, id_probs, svhn_logits, svhn_probs)
        ood_results["svhn"] = svhn_metrics
        print(f"  SVHN DS  AUPR: {svhn_metrics['dempster_shafer']['aupr']:.4f}  "
              f"Max-prob AUPR: {svhn_metrics['max_prob']['aupr']:.4f}")
    except Exception as e:
        print(f"  SVHN OOD eval skipped: {e}")

    try:
        from src.data.cifar100 import get_cifar100_loaders
        print("Running OOD detection against CIFAR-100...")
        _, cifar100_ood_loader, _, _ = get_cifar100_loaders(
            data_root=data_cfg["root"],
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 0),
        )
        c100_probs_list = [get_softmax_probs(m, cifar100_ood_loader, device)[0] for m in trained_models]
        c100_probs = torch.stack(c100_probs_list).mean(0)
        c100_logits = c100_probs.log()
        c100_metrics = evaluate_ood_split(id_logits, id_probs, c100_logits, c100_probs)
        ood_results["cifar100"] = c100_metrics
        print(f"  CIFAR-100 DS  AUPR: {c100_metrics['dempster_shafer']['aupr']:.4f}  "
              f"Max-prob AUPR: {c100_metrics['max_prob']['aupr']:.4f}")
    except Exception as e:
        print(f"  CIFAR-100 OOD eval skipped: {e}")

    # ── W&B final logging ────────────────────────────────────────────────────
    if run is not None:
        log_data: dict = {
            "ensemble/test_accuracy": ens_acc,
            "ensemble/test_nll": ens_nll,
            "ensemble/test_ece": ens_ece,
            "best/test_accuracy": ens_acc,
        }
        if "svhn" in ood_results:
            log_data["ensemble/ood_svhn_ds_aupr"] = ood_results["svhn"]["dempster_shafer"]["aupr"]
            log_data["ensemble/ood_svhn_mp_aupr"] = ood_results["svhn"]["max_prob"]["aupr"]
        if "cifar100" in ood_results:
            log_data["ensemble/ood_cifar100_ds_aupr"] = ood_results["cifar100"]["dempster_shafer"]["aupr"]
            log_data["ensemble/ood_cifar100_mp_aupr"] = ood_results["cifar100"]["max_prob"]["aupr"]
        run.log(log_data)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Deep Ensemble baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
