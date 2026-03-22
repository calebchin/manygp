import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        coeff: float = 0.95,
    ):
        super().__init__()
        self.coeff = coeff
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * self.conv(x)


class SpectralLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, coeff: float = 0.95):
        super().__init__()
        self.coeff = coeff
        self.linear = spectral_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff * self.linear(x)


class SNGPResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, coeff: float = 0.95):
        super().__init__()
        self.conv1 = SpectralConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            coeff=coeff,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SpectralConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            coeff=coeff,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                SpectralConv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    coeff=coeff,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class SNGPResNetBackbone(nn.Module):
    def __init__(self, width: int = 64, hidden_dim: int = 128, coeff: float = 0.95):
        super().__init__()
        self.stem = nn.Sequential(
            SpectralConv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1, coeff=coeff)
        self.layer2 = self._make_layer(width, width * 2, blocks=2, stride=2, coeff=coeff)
        self.layer3 = self._make_layer(width * 2, width * 4, blocks=2, stride=2, coeff=coeff)
        self.layer4 = self._make_layer(width * 4, width * 8, blocks=2, stride=2, coeff=coeff)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = SpectralLinear(width * 8, hidden_dim, bias=True, coeff=coeff)
        self.output_dim = hidden_dim

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        coeff: float,
    ) -> nn.Sequential:
        layers = [SNGPResidualBlock(in_channels, out_channels, stride=stride, coeff=coeff)]
        for _ in range(1, blocks):
            layers.append(SNGPResidualBlock(out_channels, out_channels, stride=1, coeff=coeff))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class RandomFeatureGaussianProcess(nn.Module):
    """
    Random-feature GP output layer with a diagonal-Laplace precision update.

    This follows the SNGP recipe at a practical level:
    - random Fourier features approximate an RBF GP
    - a learned linear head produces class logits
    - a running precision matrix estimates predictive variance
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_inducing: int = 1024,
        ridge_penalty: float = 1.0,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = -1.0,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing
        self.ridge_penalty = ridge_penalty
        self.feature_scale = feature_scale
        self.gp_cov_momentum = gp_cov_momentum
        self.normalize_input = normalize_input

        self.register_buffer(
            "random_weight",
            torch.randn(in_features, num_inducing) / math.sqrt(in_features),
        )
        self.register_buffer("random_bias", 2 * math.pi * torch.rand(num_inducing))
        self.beta = nn.Linear(num_inducing, out_features, bias=True)
        self.register_buffer(
            "precision_matrix",
            ridge_penalty * torch.eye(num_inducing).unsqueeze(0).repeat(out_features, 1, 1),
        )

    def reset_precision_matrix(self) -> None:
        eye = torch.eye(
            self.num_inducing,
            device=self.precision_matrix.device,
            dtype=self.precision_matrix.dtype,
        )
        self.precision_matrix.copy_(self.ridge_penalty * eye.unsqueeze(0).repeat(self.out_features, 1, 1))

    def _random_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
        projection = self.feature_scale * (x @ self.random_weight + self.random_bias)
        return math.sqrt(2.0 / self.num_inducing) * torch.cos(projection)

    @torch.no_grad()
    def _update_precision(self, phi: torch.Tensor, logits: torch.Tensor) -> None:
        probs = logits.softmax(dim=-1)
        prob_multiplier = probs * (1.0 - probs)
        batch_precision = torch.einsum("bk,bi,bj->kij", prob_multiplier, phi, phi)

        if self.gp_cov_momentum < 0:
            self.precision_matrix.add_(batch_precision)
        else:
            self.precision_matrix.mul_(self.gp_cov_momentum).add_(
                batch_precision,
                alpha=(1.0 - self.gp_cov_momentum),
            )

    def forward(
        self,
        x: torch.Tensor,
        return_cov: bool = False,
        update_precision: bool = False,
    ):
        phi = self._random_features(x)
        logits = self.beta(phi)

        if self.training and update_precision:
            self._update_precision(phi.detach(), logits.detach())

        if not return_cov:
            return logits

        precision_inv = torch.linalg.pinv(self.precision_matrix)
        variances = torch.einsum("bi,kij,bj->bk", phi, precision_inv, phi)
        variances = self.ridge_penalty * variances
        return logits, variances


def mean_field_logits(
    logits: torch.Tensor,
    variances: torch.Tensor,
    mean_field_factor: float = math.pi / 8.0,
) -> torch.Tensor:
    return logits / torch.sqrt(1.0 + mean_field_factor * variances)


class SNGPResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width: int = 64,
        hidden_dim: int = 128,
        spec_norm_bound: float = 0.95,
        num_inducing: int = 1024,
        ridge_penalty: float = 1.0,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = -1.0,
        normalize_input: bool = False,
    ):
        super().__init__()
        self.backbone = SNGPResNetBackbone(
            width=width,
            hidden_dim=hidden_dim,
            coeff=spec_norm_bound,
        )
        self.classifier = RandomFeatureGaussianProcess(
            in_features=hidden_dim,
            out_features=num_classes,
            num_inducing=num_inducing,
            ridge_penalty=ridge_penalty,
            feature_scale=feature_scale,
            gp_cov_momentum=gp_cov_momentum,
            normalize_input=normalize_input,
        )

    def reset_precision_matrix(self) -> None:
        self.classifier.reset_precision_matrix()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        return_cov: bool = False,
        update_precision: bool = False,
    ):
        features = self.encode(x)
        return self.classifier(features, return_cov=return_cov, update_precision=update_precision)
