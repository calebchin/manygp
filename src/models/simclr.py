from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import CifarResNetEncoder


class SimCLRModel(nn.Module):
    """SimCLR self-supervised model: WideResNet-28-10 encoder + 2-layer MLP projector.

    During pretraining, ``forward()`` returns L2-normalized projections used by
    NT-Xent loss.  After pretraining, only the encoder weights are kept —
    call ``load_simclr_backbone()`` to obtain a frozen ``CifarResNetEncoder``
    ready for ``FrozenBackboneSNGPClassifier``.
    """

    def __init__(
        self,
        widen_factor: int = 10,
        embedding_dim: int = 128,
        proj_hidden_dim: int = 512,
        proj_out_dim: int = 128,
    ):
        super().__init__()
        self.encoder = CifarResNetEncoder(
            widen_factor=widen_factor, embedding_dim=embedding_dim
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized projections for NT-Xent loss."""
        return F.normalize(self.projector(self.encoder(x)), dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized encoder embeddings (for kNN evaluation)."""
        return F.normalize(self.encoder(x), dim=1)


def load_simclr_backbone(
    checkpoint_path: str,
    widen_factor: int = 10,
    embedding_dim: int = 128,
    device: torch.device | None = None,
) -> CifarResNetEncoder:
    """Load a pretrained SimCLR backbone from a checkpoint.

    The checkpoint must contain an ``"encoder_state_dict"`` key produced by
    ``experiments/cifar10_simclr_pretrain.py``.

    The returned ``CifarResNetEncoder`` is frozen (``requires_grad=False``)
    and can be passed directly to ``FrozenBackboneSNGPClassifier``.
    """
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    encoder = CifarResNetEncoder(
        widen_factor=widen_factor, embedding_dim=embedding_dim
    )
    if device is not None:
        encoder = encoder.to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder
