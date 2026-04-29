import argparse
import copy
import os
import sys
import tempfile
import torch
import yaml
from pathlib import Path

from tqdm.auto import tqdm

from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_supcon_loaders
from src.models.due.wide_resnet import WideResNet
from src.models.dkl import GP, DKLModel, initial_values
from src.training.evaluate import evaluate_classifier
from src.models.resnet import CifarResNetEncoder
from src.models.sngp import WideResNet28SNGPBackbone
from src.training.contrastive import SupConLoss
from src.training.evaluate import _classification_ece


def train_dkl(
    model,
    objective,
    train_loader,
    test_loader,
    supcon_loss_fn,
    supcon_weight,
    num_epochs: int,
    lr: float,
    milestones,
    device,
    checkpoint_path,
    cfg,
    run=None,
):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    mll = VariationalELBO(objective, model.gp_layer, num_data=len(train_loader.dataset))

    best_acc = 0
    epoch_losses = []
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")

    for epoch in epochs_iter:
        model.train()
        objective.train()

        total_loss_sum = 0.0
        supcon_loss_sum = 0.0
        elbo_loss_sum = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc="Minibatch", leave=False):
            y_batch = y_batch.to(device, non_blocking=True)
            batch_size, num_views, channels, height, width = x_batch.shape
            x_batch = x_batch.to(device, non_blocking=True).view(batch_size * num_views, channels, height, width)
            ce_labels = y_batch.repeat_interleave(num_views)

            optimizer.zero_grad()
            output, features = model(x_batch, return_features=True)
            features = features.view(batch_size, num_views, -1)

            supcon_loss = supcon_loss_fn(features, y_batch)
            elbo_loss = -mll(output, ce_labels)
            total_loss = supcon_weight * supcon_loss + elbo_loss

            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            supcon_loss_sum += supcon_loss.item()
            elbo_loss_sum += elbo_loss.item()

        scheduler.step()

        avg_total_loss = total_loss_sum / len(train_loader)
        avg_supcon_loss = supcon_loss_sum / len(train_loader)
        avg_elbo_loss = elbo_loss_sum / len(train_loader)
        epoch_losses.append(avg_total_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_total_loss:.4f} | supcon: {avg_supcon_loss:.4f} | elbo: {avg_elbo_loss:.4f}")


        metric = evaluate_dkl(model, objective, test_loader, device, run)

        if run is not None:
            run.log({
                "train/total_loss": avg_total_loss,
                "train/supcon_loss": avg_supcon_loss,
                "train/elbo_loss": avg_elbo_loss,
                "train/epoch": epoch + 1,
                "train/lr": scheduler.get_last_lr()[0],
                "eval/accuracy": metric["accuracy"],
                "eval/nll": metric["nll"],
                "eval/ece": metric["ece"],
            })

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "likelihood": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, checkpoint_path)

        if metric["accuracy"] > best_acc:
            best_acc = metric["accuracy"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "likelihood": objective.state_dict(),
                "config": cfg,
            }, checkpoint_path.replace(".pt", "_best.pt"))

    return epoch_losses


def evaluate_dkl(model, objective, loader, device, run):
    model.eval()
    objective.eval()

    total_correct = 0
    total = 0
    total_loss = 0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output = model(x)

            # Match original DUE evaluation
            output = output.to_data_independent_dist()
            dist = objective(output)

            probs = dist.probs.mean(0)
            preds = probs.argmax(dim=-1)

            total_correct += (preds == y).sum().item()
            total += y.size(0)

            loss = -objective.expected_log_prob(y, output).mean()
            total_loss += loss.item() * y.size(0)

            logits = dist.logits
            ece_probs = torch.softmax(logits.mean(0), dim=-1)
            all_probs.append(ece_probs.cpu())
            all_labels.append(y.cpu())

    acc = total_correct / total
    nll = total_loss / total

    all_probs_t = torch.cat(all_probs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    ece = _classification_ece(all_probs_t, all_labels_t)

    return {"accuracy": acc, "nll": nll, "ece": ece}


def main(cfg: dict, config_path: str) -> None:
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
            project=wandb_cfg.get("project", "manygp"),
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("run_name") or None,
            config=cfg,
        )
        config_artifact = wandb.Artifact("config", type="config")
        config_artifact.add_file(config_path)
        run.log_artifact(config_artifact)

    data_cfg = cfg["data"]
    train_loader, _, val_loader, test_loader, _, memory_dataset, _, _ = get_cifar10_supcon_loaders(
        data_root=data_cfg["root"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        smoke_test=cfg["experiment"]["smoke_test"],
    )


    model_cfg = cfg["model"]
    if model_cfg.get("sngp", False):
        cnn = WideResNet28SNGPBackbone(
            widen_factor=model_cfg["widen_factor"],
            hidden_dim=model_cfg["embedding_dim"],
        ).to(device)
    else:
        cnn = CifarResNetEncoder(
            widen_factor=model_cfg["widen_factor"],
            embedding_dim=model_cfg["embedding_dim"],
        ).to(device)


    initial_inducing_points, initial_lengthscale = initial_values(
        memory_dataset, cnn, model_cfg["num_inducing_pts"]
    )

    print(f"Inducing points shape: {initial_inducing_points.shape}")

    gp = GP(
        num_outputs=model_cfg["num_classes"],
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=model_cfg.get("kernel", "RBF"),
    ).to(device)

    objective = SoftmaxLikelihood(
        num_classes=model_cfg["num_classes"],
        mixing_weights=False
    ).to(device)
    num_mc_samples = model_cfg["num_mc_samples"]
    objective.num_samples = num_mc_samples

    dkl = DKLModel(cnn, gp, objective).to(device)

    train_cfg = cfg["training"]
    supcon_loss_fn = SupConLoss(temperature=train_cfg["supcon_temperature"])
    num_epochs = 1 if cfg["experiment"]["smoke_test"] else train_cfg["num_epochs"]
    checkpoint_path = cfg["outputs"]["checkpoint_path"]
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    runtime_cfg = copy.deepcopy(cfg)
    train_dkl(
        model=dkl,
        objective=objective,
        train_loader=train_loader,
        test_loader=val_loader,
        supcon_loss_fn=supcon_loss_fn,
        supcon_weight=train_cfg["supcon_weight"],
        num_epochs=num_epochs,
        lr=train_cfg["initial_lr"],
        milestones=train_cfg["milestones"],
        device=device,
        checkpoint_path = checkpoint_path,
        cfg=runtime_cfg,
        run=run
    )

    best_ckpt = torch.load(checkpoint_path.replace(".pt", "_best.pt"), map_location=device, weights_only=False)
    dkl.load_state_dict(best_ckpt["model"])
    objective.load_state_dict(best_ckpt["likelihood"])
    metrics = evaluate_dkl(dkl, objective, test_loader, device, run)

    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test NLL: {metrics['nll']:.4f}")
    if run is not None:
        run.log({
            "test/accuracy": metrics["accuracy"],
            "test/nll": metrics["nll"],
            "test/ece": metrics["ece"],
        })

    print("\nRunning OOD + CIFAR-C evaluation...")
    from src.training.post_training_eval import run_full_ood_eval
    from src.training.ood_evaluate import collect_logits_and_probs
    from src.utils.model_loader import ModelWrapper
    wrapper = ModelWrapper(model=dkl, has_cov=False, num_mc_samples=num_mc_samples, model_type="due", likelihood=objective)
    id_logits, id_probs, _, _ = collect_logits_and_probs(wrapper, test_loader, device, num_mc_samples)
    run_full_ood_eval(
        model=dkl, has_cov=False, id_logits=id_logits, id_probs=id_probs,
        cfg=cfg, device=device, run=run, num_mc_samples=num_mc_samples, model_type="due",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 DKL experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name",
                        help="W&B run name (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
        if cfg.get("outputs", {}).get("checkpoint_path"):
            p = Path(cfg["outputs"]["checkpoint_path"])
            cfg["outputs"]["checkpoint_path"] = str(p.parent / f"seed{args.seed}" / p.name)
    if args.run_name:
        cfg.setdefault("wandb", {})["run_name"] = args.run_name

    main(cfg, config_path=args.config)