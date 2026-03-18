import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    4-block CNN feature extractor: (B, 3, 32, 32) -> (B, latent_dim).

    Designed for CIFAR-10 sized images (32x32). Can be used as a backbone for
    any downstream task (classification, GP-based models, etc.).

    Workflow:
    - Pretrain with `forward_cls` (uses the classification head) via cross-entropy.
    - Call `freeze_backbone()` to lock the backbone and projector weights.
    - Use `forward` to extract latent features for a downstream model.
    """

    def __init__(self, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
            # Block 4: 128x4x4 -> 256x2x2
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # (B, 256)
        )
        self.projector = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # Classification head — used only during pretraining, then frozen/discarded
        self.head = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent feature vectors of shape (B, latent_dim)."""
        return self.projector(self.backbone(x))

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits of shape (B, num_classes). Used during pretraining."""
        return self.head(self.forward(x))

    def freeze_backbone(self) -> None:
        """Freeze backbone and projector parameters (keeps head trainable if needed)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = False
