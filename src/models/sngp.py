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
        optimize_length_scale: bool = False,
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
        self.optimize_length_scale = optimize_length_scale

        # For normalized_rbf: keep random_weight as N(0,I) and apply length scale
        # dynamically in _random_features (matches GPyTorch RFFKernel._featurize pattern).
        # For legacy: scale by 1/sqrt(d) at init as before.
        if kernel_type == "normalized_rbf":
            random_weight = torch.randn(in_features, num_inducing)
        else:
            random_weight = torch.randn(in_features, num_inducing) / math.sqrt(in_features)

        self.register_buffer("random_weight", random_weight)
        self.register_buffer("random_bias", 2 * math.pi * torch.rand(num_inducing))

        # log_length_scale: learnable parameter when optimize_length_scale=True, buffer otherwise.
        # Stored in log-space so exp() always gives a positive value.
        log_ls = torch.tensor(math.log(max(length_scale, 1e-12)))
        if optimize_length_scale and kernel_type == "normalized_rbf":
            self.log_length_scale = nn.Parameter(log_ls)
        else:
            self.register_buffer("log_length_scale", log_ls)

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
            # Divide by length scale dynamically so gradients can flow through log_length_scale
            # when optimize_length_scale=True. Mirrors GPyTorch RFFKernel._featurize().
            l = self.log_length_scale.exp()
            projection = x @ (self.random_weight / l) + self.random_bias
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

    def _compute_precision_grad(self, phi: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Differentiable re-estimate of the precision matrix on the given batch.

        Unlike _update_precision (which is @no_grad and accumulates into a buffer),
        this version stays in the autograd graph so gradients can flow through it
        to log_length_scale during the MML optimization step.

        Returns shape: (out_features, num_inducing, num_inducing)
        """
        probs = logits.softmax(dim=-1)
        prob_mult = probs * (1.0 - probs)
        precision = torch.einsum("bk,bi,bj->kij", prob_mult, phi, phi)
        eye = torch.eye(self.num_inducing, device=phi.device, dtype=phi.dtype).unsqueeze(0)
        return precision + self.ridge_penalty * eye

    def compute_laplace_log_mml(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Laplace approximation to the log marginal likelihood w.r.t. log_length_scale.

            log p(Y | X, l) ≈ log p(Y | θ_MAP) − ½ Σ_k log|P_k(l)|

        Returns the *negative* log MML (scalar) so it can be minimized directly.
        Gradients flow through log_length_scale via _random_features.
        """
        phi = self._random_features(x)   # differentiable w.r.t. log_length_scale
        logits = self.beta(phi)

        ce_term = -F.cross_entropy(logits, labels, reduction="sum")
        precision = self._compute_precision_grad(phi, logits)  # [K, D, D]

        # log|P_k| via Cholesky — numerically stable and differentiable
        try:
            L = torch.linalg.cholesky(precision)
            log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # [K]
        except torch.linalg.LinAlgError:
            sign, log_det = torch.linalg.slogdet(precision)
            log_det = log_det * sign.clamp(min=0)

        log_mml = ce_term - 0.5 * log_det.sum()
        return -log_mml  # return negative for minimization

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
        optimize_length_scale: bool = False,
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
            optimize_length_scale=optimize_length_scale,
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
