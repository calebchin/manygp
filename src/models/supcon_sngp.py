from __future__ import annotations

import torch
import torch.nn as nn

from .resnet import CifarResNetEncoder
from .sngp import RandomFeatureGaussianProcess


class CifarResNetSupConSNGPClassifier(nn.Module):
    """
    Wide ResNet-28-10 encoder with an optional MLP head and SNGP output layer.

    The GP input features are exposed so supervised contrastive loss can be
    applied directly on the deterministic representation before the GP layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        widen_factor: int = 10,
        hidden_dims: list[int] | tuple[int, ...] = (),
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
        self.encoder = CifarResNetEncoder(widen_factor=widen_factor, embedding_dim=embedding_dim)
        self.encoder_dim = self.encoder.output_dim

        layers = []
        in_dim = self.encoder_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
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
        return_features: bool = False,
    ):
        features = self.encode(x)
        outputs = self.classifier(features, return_cov=return_cov, update_precision=update_precision)
        if return_features:
            return outputs, features
        return outputs
