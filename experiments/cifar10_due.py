import argparse
import os
import sys
import tempfile
import torch
import yaml

from tqdm.auto import tqdm

from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.mlls import VariationalELBO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cifar10 import get_cifar10_loaders
from src.models.due.wide_resnet import WideResNet
from src.models.dkl import GP, DKLModel, initial_values
from src.training.evaluate import evaluate_classifier


def train_dkl(
    model,
    objective,
    train_loader,
    test_loader,
    num_epochs: int,
    lr: float,
    milestones,
    device,
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

    epoch_losses = []
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for epoch in epochs_iter:
        model.train()
        objective.train()

        total_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc="Minibatch", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # IMPORTANT: model outputs GP distribution (NOT softmax)
            output = model(x_batch)

            loss = -mll(output, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")


        metric = evaluate_dkl(model, objective, test_loader, device, run)

        if run is not None:
            run.log({
                "train/loss": avg_loss,
                "train/epoch": epoch + 1,
                "train/lr": scheduler.get_last_lr()[0],
                "eval/acc": metric["acc"],
                "eval/nll": metric["nll"],
            })

    return epoch_losses


def evaluate_dkl(model, objective, loader, device, run):
    model.eval()
    objective.eval()

    total_correct = 0
    total = 0
    total_loss = 0

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

    acc = total_correct / total
    nll = total_loss / total

    return {"accuracy": acc, "nll": nll}


def main(cfg: dict, config_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    train_loader, test_loader, dataset_train, dataset_test = get_cifar10_loaders(
        data_root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        smoke_test=cfg["experiment"]["smoke_test"],
    )

    cnn_cfg = cfg["cnn"]

    cnn = WideResNet(
        input_size=32,
        spectral_conv=cnn_cfg["spectral_conv"],
        spectral_bn=cnn_cfg["spectral_bn"],
    ).to(device)

    dkl_cfg = cfg["dkl"]

    # ? Correct inducing initialization
    initial_inducing_points, initial_lengthscale = initial_values(
        dataset_train, cnn, dkl_cfg["num_inducing_pts"]
    )

    print(f"Inducing points shape: {initial_inducing_points.shape}")

    gp = GP(
        num_outputs=cnn_cfg["num_classes"],
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=dkl_cfg.get("kernel", "RBF"),
    ).to(device)

    # likelihood (kept separate, like original)
    objective = SoftmaxLikelihood(
        num_classes=cnn_cfg["num_classes"],
        mixing_weights=False
    ).to(device)

    # ?? IMPORTANT: control MC samples
    objective.num_samples = 5

    # model = feature extractor + GP ONLY
    dkl = DKLModel(cnn, gp, objective).to(device)

    train_cfg = cfg["training"]

    num_epochs = 1 if cfg["experiment"]["smoke_test"] else train_cfg["num_epochs"]

    train_dkl(
        model=dkl,
        objective=objective,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        lr=train_cfg["initial_lr"],
        milestones=train_cfg["milestones"],
        device=device,
    )

    metrics = evaluate_dkl(dkl, objective, test_loader, device)

    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test NLL: {metrics['nll']:.4f}")

    # Save checkpoint
    torch.save({"dkl": dkl.state_dict()}, "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 DKL experiment")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, config_path=args.config)