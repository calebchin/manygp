from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class SupConLoss(torch.nn.Module):
    """Supervised contrastive loss from https://arxiv.org/abs/2004.11362."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape [batch, views, dim], got {features.shape}")

        device = features.device
        _, num_views, _ = features.shape
        features = F.normalize(features, dim=-1)
        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)

        logits = contrast_features @ contrast_features.T
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(num_views, num_views)

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0.0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_anchor = mask.sum(dim=1)
        valid = positives_per_anchor > 0
        mean_log_prob_pos = torch.zeros_like(positives_per_anchor)
        mean_log_prob_pos[valid] = (mask[valid] * log_prob[valid]).sum(dim=1) / positives_per_anchor[valid]

        loss = -mean_log_prob_pos[valid].mean()
        if not torch.isfinite(loss):
            raise RuntimeError("SupCon loss became non-finite")
        return loss

class NoAugSupConLoss(torch.nn.Module):
    """Supervised contrastive loss from https://arxiv.org/abs/2004.11362."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected features with shape [batch, dim], got {features.shape}")

        device = features.device
        features = F.normalize(features, dim=1)

        logits = features @ features.T
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0.0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_anchor = mask.sum(dim=1)
        valid = positives_per_anchor > 0
        mean_log_prob_pos = torch.zeros_like(positives_per_anchor)
        mean_log_prob_pos[valid] = (mask[valid] * log_prob[valid]).sum(dim=1) / positives_per_anchor[valid]

        loss = -mean_log_prob_pos[valid].mean()
        if not torch.isfinite(loss):
            raise RuntimeError("SupCon loss became non-finite")
        return loss

def train_supcon(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: SupConLoss,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    run=None,
    log_every_steps: int | None = None,
    global_step: int = 0,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"SupCon Epoch {epoch}",
        leave=False,
        disable=not show_progress,
    )
    for views, labels in progress:
        labels = labels.to(device, non_blocking=True)
        batch_size, num_views, channels, height, width = views.shape
        views = views.to(device, non_blocking=True).view(batch_size * num_views, channels, height, width)

        optimizer.zero_grad(set_to_none=True)
        projections = model(views).view(batch_size, num_views, -1)
        loss = loss_fn(projections, labels)
        loss.backward()
        optimizer.step()

        global_step += 1
        running_loss += loss.item()
        progress.set_postfix(loss=loss.item())
        if run is not None and log_every_steps is not None and log_every_steps > 0:
            if global_step % log_every_steps == 0:
                run.log({
                    "train/step_loss": loss.item(),
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/lr_step": optimizer.param_groups[0]["lr"],
                })

    if scheduler is not None:
        scheduler.step()

    return running_loss / len(train_loader), global_step


@torch.no_grad()
def compute_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_embeddings = model.encode(images).cpu()
        embeddings.append(batch_embeddings)
        labels.append(batch_labels.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def knn_accuracy(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.1,
    chunk_size: int = 512,
) -> float:
    num_classes = int(train_labels.max().item()) + 1
    train_embeddings = F.normalize(train_embeddings, dim=1)
    query_embeddings = F.normalize(query_embeddings, dim=1)

    total_correct = 0
    for start in range(0, query_embeddings.size(0), chunk_size):
        query_chunk = query_embeddings[start : start + chunk_size]
        label_chunk = query_labels[start : start + chunk_size]

        similarities = query_chunk @ train_embeddings.T
        topk_similarities, topk_indices = similarities.topk(k=min(k, train_embeddings.size(0)), dim=1)
        topk_labels = train_labels[topk_indices]

        weights = torch.exp(topk_similarities / temperature)
        class_scores = torch.zeros(query_chunk.size(0), num_classes)
        class_scores.scatter_add_(1, topk_labels, weights)
        predictions = class_scores.argmax(dim=1)
        total_correct += (predictions == label_chunk).sum().item()

    return total_correct / query_embeddings.size(0)


def evaluate_knn(
    model: torch.nn.Module,
    memory_loader,
    val_loader,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.1,
) -> Dict[str, float]:
    train_embeddings, train_labels = compute_embeddings(model, memory_loader, device)
    val_embeddings, val_labels = compute_embeddings(model, val_loader, device)
    accuracy = knn_accuracy(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=val_embeddings,
        query_labels=val_labels,
        k=k,
        temperature=temperature,
    )
    return {"knn_accuracy": accuracy}
