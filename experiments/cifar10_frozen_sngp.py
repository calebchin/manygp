"""
CIFAR-10 frozen-backbone SNGP training.

Loads a pretrained ResNet encoder from a checkpoint, freezes it, and trains a
configurable shallow SNGP head on top for NLL-focused comparison.

Usage:
    python experiments/cifar10_frozen_sngp.py --config configs/cifar10_frozen_sngp.yaml
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
from src.models.frozen_sngp import FrozenBackboneSNGPClassifier, load_frozen_resnet_encoder
from src.models.sngp import laplace_predictive_probs
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

    def is_better(a: float, b: float) -> bool:
        return a < b if lower_is_better else a > b

    if len(saved_checkpoints) < top_k:
        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
    else:
        worst_checkpoint = (
            max(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
            if lower_is_better
            else min(saved_checkpoints, key=lambda item: (item["metric"], -item["epoch"]))
        )
        if not is_better(metric_value, worst_checkpoint["metric"]):
            return

        torch.save(state, candidate_path)
        saved_checkpoints.append({"metric": metric_value, "path": candidate_path, "epoch": epoch})
        worst_checkpoint["path"].unlink(missing_ok=True)
        saved_checkpoints.remove(worst_checkpoint)

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
    model.train()
    model.reset_precision_matrix()
    running_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(loader, desc=f"Frozen SNGP Epoch {epoch}", leave=False, disable=not show_progress)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
def evaluate_frozen_sngp(
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

    return {
        "loss": running_loss / len(loader),
        "accuracy": total_correct / total_examples,
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
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=smoke_test,
    )
    print(f"Train size: {len(train_dataset)}, test size: {len(test_dataset)}")

    backbone_cfg = cfg["backbone"]
    frozen_encoder = load_frozen_resnet_encoder(
        checkpoint_path=backbone_cfg["checkpoint_path"],
        width=backbone_cfg["width"],
        embedding_dim=backbone_cfg["embedding_dim"],
        device=device,
    )

    model_cfg = cfg["model"]
    model = FrozenBackboneSNGPClassifier(
        encoder=frozen_encoder,
        num_classes=model_cfg["num_classes"],
        hidden_dims=model_cfg["hidden_dims"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
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
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1 if smoke_test else train_cfg["epochs"],
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_test_nll = float("inf")
    num_epochs = 1 if smoke_test else train_cfg["epochs"]
    eval_interval = 1 if smoke_test else train_cfg.get("eval_interval", 1)
    log_every_steps = train_cfg.get("log_every_steps", None)
    num_mc_samples = train_cfg.get("num_mc_samples", 10)
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
        test_loss = None
        test_acc = None
        test_nll = None
        if should_evaluate:
            metrics = evaluate_frozen_sngp(
                model=model,
                loader=test_loader,
                device=device,
                num_mc_samples=num_mc_samples,
            )
            test_loss = metrics["loss"]
            test_acc = metrics["accuracy"]
            test_nll = metrics["nll"]
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc * 100:.2f}% | "
                f"Test NLL: {test_nll:.4f}"
            )
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc * 100:.2f}%",
                test_nll=f"{test_nll:.4f}",
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
            if test_loss is not None and test_acc is not None and test_nll is not None:
                log_data["test/loss"] = test_loss
                log_data["test/accuracy"] = test_acc
                log_data["test/nll"] = test_nll
            run.log(log_data)

        if test_nll is None:
            continue

        if test_nll < best_test_nll:
            best_test_nll = test_nll
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "test_nll": test_nll,
            "config": cfg,
        }
        if checkpoint_path:
            update_topk_checkpoints(
                saved_checkpoints=saved_checkpoints,
                top_k=top_k,
                checkpoint_path=checkpoint_path,
                state=checkpoint_state,
                metric_name="nll",
                metric_value=test_nll,
                epoch=epoch,
                lower_is_better=True,
            )

    if saved_checkpoints:
        print("Saved top checkpoints:")
        for checkpoint in saved_checkpoints:
            print(
                f"  epoch {checkpoint['epoch']:3d} | "
                f"test nll {checkpoint['metric']:.4f} | "
                f"{checkpoint['path']}"
            )

    if best_test_nll < float("inf"):
        print(f"Best test NLL: {best_test_nll:.4f}")
    else:
        print("Best test NLL: not evaluated")

    if run is not None:
        if best_test_nll < float("inf"):
            run.log({"best/test_nll": best_test_nll})
        if saved_checkpoints:
            import wandb

            artifact = wandb.Artifact("cifar10_frozen_sngp_best_model", type="model")
            artifact.add_file(str(saved_checkpoints[0]["path"]), name=saved_checkpoints[0]["path"].name)
            run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 frozen-backbone shallow SNGP")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
