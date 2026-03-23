from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import CifarResNetEncoder
from .sngp import RandomFeatureGaussianProcess, SpectralLinear


def _extract_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value
    if not encoder_state:
        raise ValueError("Checkpoint does not contain encoder.* weights")
    return encoder_state


def load_frozen_resnet_encoder(
    checkpoint_path: str,
    width: int,
    embedding_dim: int,
    device: torch.device,
) -> CifarResNetEncoder:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    encoder = CifarResNetEncoder(width=width, embedding_dim=embedding_dim).to(device)
    encoder.load_state_dict(_extract_encoder_state_dict(state_dict))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


class FrozenBackboneSNGPClassifier(nn.Module):
    """
    Frozen pretrained ResNet encoder with a configurable shallow SNGP head.

    The trainable part is:
    frozen encoder -> configurable shallow MLP -> random-feature GP output layer
    """

    def __init__(
        self,
        encoder: CifarResNetEncoder,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...],
        spec_norm_bound: float = 0.95,
        dropout_rate: float = 0.0,
        num_inducing: int = 1024,
        ridge_penalty: float = 1.0,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = -1.0,
        normalize_input: bool = False,
        kernel_type: str = "normalized_rbf",
        input_normalization: str = "l2",
        kernel_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder.output_dim

        layers = []
        in_dim = self.encoder_dim
        for hidden_dim in hidden_dims:
            layers.append(SpectralLinear(in_dim, hidden_dim, coeff=spec_norm_bound))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        self.head = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = RandomFeatureGaussianProcess(
            in_features=in_dim,
            out_features=num_classes,
            num_inducing=num_inducing,
            ridge_penalty=ridge_penalty,
            feature_scale=feature_scale,
            gp_cov_momentum=gp_cov_momentum,
            normalize_input=normalize_input,
            kernel_type=kernel_type,
            input_normalization=input_normalization,
            kernel_scale=kernel_scale,
            length_scale=length_scale,
        )

    def reset_precision_matrix(self) -> None:
        self.classifier.reset_precision_matrix()

    @torch.no_grad()
    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_backbone(x)
        return self.head(features)

    def forward(
        self,
        x: torch.Tensor,
        return_cov: bool = False,
        update_precision: bool = False,
    ):
        features = self.encode(x)
        return self.classifier(features, return_cov=return_cov, update_precision=update_precision)
