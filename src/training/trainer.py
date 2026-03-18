from typing import List, Optional

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from tqdm.auto import tqdm


def init_inducing_points_kmeans(data_tensor: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Initialise inducing points via k-means clustering.

    Args:
        data_tensor: (N, D) CPU tensor of data points.
        n_points:    Number of inducing points to produce.

    Returns:
        Tensor of shape (n_points, D).
    """
    data_np = data_tensor.cpu().numpy()
    init = data_np[np.random.permutation(len(data_np))[:n_points]]
    centroids, _ = kmeans2(data_np, init, minit="matrix")
    return torch.tensor(centroids, dtype=data_tensor.dtype)


def extract_cnn_features(cnn, loader, n_samples: int, device) -> torch.Tensor:
    """
    Extract up to n_samples feature vectors from a frozen CNN.

    Args:
        cnn:      Feature extractor (nn.Module, kept in eval mode).
        loader:   DataLoader yielding (x_batch, y_batch).
        n_samples: Maximum number of samples to collect.
        device:   torch.device

    Returns:
        Tensor of shape (n_samples, latent_dim).
    """
    cnn.eval()
    chunks = []
    collected = 0
    with torch.no_grad():
        for x_batch, _ in loader:
            feats = cnn(x_batch.to(device)).cpu()
            chunks.append(feats)
            collected += feats.size(0)
            if collected >= n_samples:
                break
    return torch.cat(chunks)[:n_samples]


def pretrain_cnn(
    cnn,
    train_loader,
    epochs: int,
    device,
    lr: float = 1e-2,
    milestones: Optional[List[int]] = None,
    run=None,
) -> None:
    """
    Pretrain a CNNFeatureExtractor with cross-entropy loss.

    Freezes the backbone and projector by calling cnn.freeze_backbone() when done.

    Args:
        cnn:          CNNFeatureExtractor instance.
        train_loader: DataLoader yielding (x_batch, y_batch).
        epochs:       Number of training epochs.
        device:       torch.device
        lr:           Initial learning rate for Adam.
        milestones:   Epoch milestones for MultiStepLR (gamma=0.1).
        run:          Optional W&B run object. If provided, logs per-epoch metrics.
    """
    if milestones is None:
        milestones = [10, 15]

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Pretraining CNN with cross-entropy...")
    for epoch in tqdm(range(epochs), desc="Pretrain Epoch"):
        cnn.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(cnn.forward_cls(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"  Pretrain Epoch {epoch + 1:2d}/{epochs} | CE Loss: {avg_loss:.4f}")
        if run is not None:
            run.log({"pretrain/ce_loss": avg_loss, "pretrain/epoch": epoch + 1})

    cnn.freeze_backbone()


def train_dspp(
    model,
    objective,
    train_loader,
    num_epochs: int,
    lr: float,
    milestones: List[int],
    device,
    cnn=None,
    run=None,
) -> List[float]:
    """
    Train a DSPP model with Adam + MultiStepLR.

    Fixes the tqdm.notebook crash from the original dspp.ipynb by using
    tqdm.auto instead.

    Args:
        model:        DSPP model (TwoLayerDSPPClassifier or TwoLayerDSPP).
        objective:    DeepPredictiveLogLikelihood instance.
        train_loader: DataLoader yielding (x_batch, y_batch).
        num_epochs:   Number of training epochs.
        lr:           Initial learning rate.
        milestones:   Epoch milestones for MultiStepLR (gamma=0.1).
        device:       torch.device
        cnn:          Optional feature extractor (CIFAR-10 only).
                      If provided, kept in eval mode; features are extracted in-loop.
        run:          Optional W&B run object. If provided, logs per-epoch metrics.

    Returns:
        List of per-epoch average losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    epoch_losses = []
    epochs_iter = tqdm(range(num_epochs), desc="Epoch")
    for epoch in epochs_iter:
        if cnn is not None:
            cnn.eval()
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc="Minibatch", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            inputs = cnn(x_batch) if cnn is not None else x_batch
            output = model(inputs)
            loss = -objective(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        epochs_iter.set_postfix(loss=avg_loss)
        print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.4f}")
        if run is not None:
            run.log({
                "train/loss": avg_loss,
                "train/epoch": epoch + 1,
                "train/lr": scheduler.get_last_lr()[0],
            })

    return epoch_losses
