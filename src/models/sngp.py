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


class WideResNetPreActBlock(nn.Module):
    """
    Pre-activation residual block for Wide ResNet with spectral normalization.

    Order: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, with shortcut on raw input.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, coeff: float = 0.95):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = SpectralConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, coeff=coeff
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = SpectralConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = SpectralConv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, coeff=coeff
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class WideResNet28SNGPBackbone(nn.Module):
    """
    Wide ResNet-28-10 backbone with spectral normalization for SNGP.

    Architecture (Liu et al., 2020):
      - Depth 28 = 6 * 4 + 4  (4 blocks per group, 3 groups)
      - Widen factor 10: channels [16*10, 32*10, 64*10] = [160, 320, 640]
      - Pre-activation residual blocks
      - Spectral normalization on all Conv and Linear layers
      - Final projection: 640 -> hidden_dim (for GP input)
    """

    _N_BLOCKS = 4
    _BASE_CHANNELS = 16

    def __init__(self, widen_factor: int = 10, hidden_dim: int = 128, coeff: float = 0.95):
        super().__init__()
        base = self._BASE_CHANNELS
        w = [base * widen_factor, 2 * base * widen_factor, 4 * base * widen_factor]

        self.stem = SpectralConv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff)
        self.group1 = self._make_group(base,    w[0], self._N_BLOCKS, stride=1, coeff=coeff)
        self.group2 = self._make_group(w[0],    w[1], self._N_BLOCKS, stride=2, coeff=coeff)
        self.group3 = self._make_group(w[1],    w[2], self._N_BLOCKS, stride=2, coeff=coeff)
        self.bn_final = nn.BatchNorm2d(w[2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = SpectralLinear(w[2], hidden_dim, bias=True, coeff=coeff)
        self.output_dim = hidden_dim

    def _make_group(
        self, in_channels: int, out_channels: int, n_blocks: int, stride: int, coeff: float
    ) -> nn.Sequential:
        blocks = [WideResNetPreActBlock(in_channels, out_channels, stride=stride, coeff=coeff)]
        for _ in range(1, n_blocks):
            blocks.append(WideResNetPreActBlock(out_channels, out_channels, stride=1, coeff=coeff))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn_final(x), inplace=True)
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
        kernel_type: str = "legacy",
        input_normalization: str | None = None,
        kernel_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_inducing = num_inducing
        self.ridge_penalty = ridge_penalty
        self.feature_scale = feature_scale
        self.gp_cov_momentum = gp_cov_momentum
        self.normalize_input = normalize_input
        self.kernel_type = kernel_type
        self.input_normalization = (
            input_normalization if input_normalization is not None else ("layer_norm" if normalize_input else "none")
        )
        self.kernel_scale = kernel_scale
        self.length_scale = length_scale

        if kernel_type == "normalized_rbf":
            random_weight = torch.randn(in_features, num_inducing) / max(length_scale, 1e-12)
        else:
            random_weight = torch.randn(in_features, num_inducing) / math.sqrt(in_features)

        self.register_buffer("random_weight", random_weight)
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

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "layer_norm":
            x = F.layer_norm(x, normalized_shape=(x.shape[-1],))
        elif self.input_normalization == "l2":
            x = F.normalize(x, p=2, dim=-1)
        return x

    def _random_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_features(x)
        if self.kernel_type == "normalized_rbf":
            projection = x @ self.random_weight + self.random_bias
            return self.kernel_scale * math.sqrt(2.0 / self.num_inducing) * torch.cos(projection)

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


def laplace_predictive_probs(
    logits: torch.Tensor,
    variances: torch.Tensor,
    num_mc_samples: int = 10,
) -> torch.Tensor:
    std = torch.sqrt(torch.clamp(variances, min=1e-12))
    noise = torch.randn(
        num_mc_samples,
        *logits.shape,
        device=logits.device,
        dtype=logits.dtype,
    )
    sampled_logits = logits.unsqueeze(0) + noise * std.unsqueeze(0)
    return sampled_logits.softmax(dim=-1).mean(dim=0)


class SNGPResNetClassifier(nn.Module):
    """
    Wide ResNet-28-10 with spectral normalization + Random Feature GP head.

    Implements the SNGP model from Liu et al. (2020) for CIFAR-scale inputs.
    """

    def __init__(
        self,
        num_classes: int = 10,
        widen_factor: int = 10,
        hidden_dim: int = 128,
        spec_norm_bound: float = 0.95,
        num_inducing: int = 1024,
        ridge_penalty: float = 1.0,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = -1.0,
        normalize_input: bool = False,
        kernel_type: str = "legacy",
        input_normalization: str | None = None,
        kernel_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = WideResNet28SNGPBackbone(
            widen_factor=widen_factor,
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
            kernel_type=kernel_type,
            input_normalization=input_normalization,
            kernel_scale=kernel_scale,
            length_scale=length_scale,
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
        return_features: bool = False,
    ):
        features = self.encode(x)
        outputs = self.classifier(features, return_cov=return_cov, update_precision=update_precision)
        if return_features:
            return outputs, features
        return outputs


class VGGStyleSNGPBackbone(nn.Module):
    """
    VGG-style CNN backbone with spectral normalization for SNGP.

    8 conv layers in 4 blocks (each followed by MaxPool), all spectrally
    normalized to bound the Lipschitz constant without skip connections.
    Simpler Lipschitz analysis than ResNet: no shortcut paths to worry about.

    Architecture for 32x32 CIFAR inputs:
      Block 1: Conv(3→64)×2   + MaxPool → 16×16
      Block 2: Conv(64→128)×2 + MaxPool → 8×8
      Block 3: Conv(128→256)×2 + MaxPool → 4×4
      Block 4: Conv(256→512)   + AdaptiveAvgPool → 1×1
      Projection: Linear(512 → embedding_dim)
    """

    def __init__(self, embedding_dim: int = 128, coeff: float = 0.95):
        super().__init__()
        c = coeff
        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            SpectralConv2d(3,   64,  3, padding=1, coeff=c), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            SpectralConv2d(64,  64,  3, padding=1, coeff=c), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 16×16 → 8×8
            SpectralConv2d(64,  128, 3, padding=1, coeff=c), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            SpectralConv2d(128, 128, 3, padding=1, coeff=c), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 8×8 → 4×4
            SpectralConv2d(128, 256, 3, padding=1, coeff=c), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            SpectralConv2d(256, 256, 3, padding=1, coeff=c), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 4×4 → 1×1
            SpectralConv2d(256, 512, 3, padding=1, coeff=c), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = SpectralLinear(512, embedding_dim, bias=True, coeff=coeff)
        self.output_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class CNNSupConSNGPClassifier(nn.Module):
    """
    VGG-style spectrally-normalized CNN backbone + RFGP head.

    Drop-in replacement for CifarResNetSupConSNGPClassifier — same interface,
    same forward() signature including return_features=True for MS/SupCon loss.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 10,
        spec_norm_bound: float = 0.95,
        num_inducing: int = 1024,
        ridge_penalty: float = 1e-3,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = 0.999,
        normalize_input: bool = False,
        kernel_type: str = "normalized_rbf",
        input_normalization: str = "l2",
        kernel_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = VGGStyleSNGPBackbone(
            embedding_dim=embedding_dim, coeff=spec_norm_bound
        )
        self.classifier = RandomFeatureGaussianProcess(
            in_features=embedding_dim,
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        return_cov: bool = False,
        update_precision: bool = False,
        return_features: bool = False,
    ):
        features = self.encode(x)
        outputs = self.classifier(
            features, return_cov=return_cov, update_precision=update_precision
        )
        if return_features:
            return outputs, features
        return outputs


class WideResNetPreActBlockNoSkip(nn.Module):
    """
    Pre-activation WRN block with spectral normalization but NO residual skip.

    Same conv structure as WideResNetPreActBlock, just removes the `+ shortcut(x)`
    addition. This turns WRN-28-10 into a plain deep network with the same
    depth, width, and Lipschitz constraint — isolating the effect of skip connections.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, coeff: float = 0.95):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = SpectralConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, coeff=coeff
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = SpectralConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out  # no residual addition


class WideResNet28NoSkipBackbone(nn.Module):
    """
    WRN-28-10 with spectral normalization but WITHOUT skip connections.

    Identical channel structure and depth to WideResNet28SNGPBackbone:
      channels [160, 320, 640], 4 blocks per group, 3 groups.
    The only difference: residual additions are removed, making it a plain
    28-layer deep network. Useful for ablating the role of skip connections
    independently from depth, width, and spectral normalization.
    """

    _N_BLOCKS = 4
    _BASE_CHANNELS = 16

    def __init__(self, widen_factor: int = 10, hidden_dim: int = 128, coeff: float = 0.95):
        super().__init__()
        base = self._BASE_CHANNELS
        w = [base * widen_factor, 2 * base * widen_factor, 4 * base * widen_factor]

        self.stem = SpectralConv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False, coeff=coeff)
        self.group1 = self._make_group(base,    w[0], self._N_BLOCKS, stride=1, coeff=coeff)
        self.group2 = self._make_group(w[0],    w[1], self._N_BLOCKS, stride=2, coeff=coeff)
        self.group3 = self._make_group(w[1],    w[2], self._N_BLOCKS, stride=2, coeff=coeff)
        self.bn_final = nn.BatchNorm2d(w[2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = SpectralLinear(w[2], hidden_dim, bias=True, coeff=coeff)
        self.output_dim = hidden_dim

    def _make_group(self, in_ch, out_ch, n_blocks, stride, coeff):
        blocks = [WideResNetPreActBlockNoSkip(in_ch, out_ch, stride=stride, coeff=coeff)]
        for _ in range(1, n_blocks):
            blocks.append(WideResNetPreActBlockNoSkip(out_ch, out_ch, stride=1, coeff=coeff))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn_final(x), inplace=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class WRNNoSkipSupConSNGPClassifier(nn.Module):
    """
    WRN-28-10 depth/width, spectral norm, NO skip connections + RFGP head.

    Uses WideResNet28NoSkipBackbone. Same interface as CifarResNetSupConSNGPClassifier
    including return_features=True for MS/SupCon losses.
    """

    def __init__(
        self,
        num_classes: int = 10,
        widen_factor: int = 10,
        hidden_dim: int = 128,
        spec_norm_bound: float = 6.0,
        num_inducing: int = 1024,
        ridge_penalty: float = 1e-3,
        feature_scale: float = 2.0,
        gp_cov_momentum: float = 0.999,
        normalize_input: bool = False,
        kernel_type: str = "normalized_rbf",
        input_normalization: str = "l2",
        kernel_scale: float = 1.0,
        length_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = WideResNet28NoSkipBackbone(
            widen_factor=widen_factor, hidden_dim=hidden_dim, coeff=spec_norm_bound
        )
        self.classifier = RandomFeatureGaussianProcess(
            in_features=hidden_dim,
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
        return_cov: bool = False,
        update_precision: bool = False,
        return_features: bool = False,
    ):
        features = self.encode(x)
        outputs = self.classifier(
            features, return_cov=return_cov, update_precision=update_precision
        )
        if return_features:
            return outputs, features
        return outputs
