"""
Unified model loader for OOD evaluation.

Reads checkpoint['config']['experiment']['name'] to auto-detect model type,
reconstructs the model, loads weights, and returns a ModelWrapper with a
consistent predict_logits / predict_probs interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ModelWrapper:
    """
    Unified inference interface for all supported model types:
      - SNGPResNetClassifier          (cifar10_sngp, cifar10_sngp_augmented)
      - CifarResNetClassifier         (cifar10_classifier)
      - FrozenBackboneSNGPClassifier  (cifar10_frozen_sngp)
      - CifarResNetSupConSNGPClassifier (cifar10_supcon_sngp)
      - DKL                             (due)
    """

    def __init__(
        self,
        model: nn.Module,
        has_cov: bool,
        num_mc_samples: int = 10,
        model_type: str = "",
        likelihood: nn.Module | None = None,
    ):
        self.model = model
        self.has_cov = has_cov
        self.num_mc_samples = num_mc_samples
        self.model_type = model_type
        self.likelihood = likelihood

        if self.likelihood is not None:
            self.likelihood.num_samples = num_mc_samples

    @torch.no_grad()
    def predict_logits(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits, variances). variances is None for plain classifiers."""
        if self.model_type == "due":
            raise RuntimeError("Use predict() for DUE")
        if self.has_cov:
            logits, variances = self.model(x, return_cov=True)
            return logits, variances
        else:
            logits = self.model(x)
            return logits, None

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (N, K) softmax probabilities (Laplace for SNGP variants)."""
        if self.model_type == "due":
            output = self.model(x)
            output = output.to_data_independent_dist()

            dist = self.model.likelihood(output)
            probs = dist.probs.mean(0)

            logits = torch.log(probs + 1e-8)
            uncertainty = -(probs * probs.log()).sum(1)
            return uncertainty, probs

        from src.models.sngp import laplace_predictive_probs

        logits, variances = self.predict_logits(x)
        if self.has_cov and variances is not None:
            return laplace_predictive_probs(logits, variances, num_mc_samples=self.num_mc_samples)
        return F.softmax(logits, dim=-1)


def _build_due(model_cfg: dict, device: torch.device):
    from gpytorch.likelihoods import SoftmaxLikelihood
    from src.models.dkl import GP, DKLModel
    from src.models.resnet import CifarResNetEncoder

    cnn = CifarResNetEncoder(
        widen_factor=model_cfg["widen_factor"],
        embedding_dim=model_cfg["embedding_dim"],
    ).to(device)

    gp = GP(
        num_outputs=model_cfg["num_classes"],
        initial_lengthscale=0.0,
        initial_inducing_points=torch.zeros(
            model_cfg["num_inducing"], model_cfg["feature_dim"]
        ).to(device),
        kernel=model_cfg.get("kernel", "RBF"),
    ).to(device)

    model = DKLModel(cnn, gp).to(device)

    likelihood = SoftmaxLikelihood(
        num_classes=model_cfg["num_classes"],
        mixing_weights=False,
    ).to(device)

    return model, likelihood, False


def _build_sngp(model_cfg: dict, device: torch.device) -> tuple[nn.Module, bool]:
    from src.models.sngp import SNGPResNetClassifier

    model = SNGPResNetClassifier(
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dim=model_cfg["hidden_dim"],
        spec_norm_bound=model_cfg["spec_norm_bound"],
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg["normalize_input"],
    ).to(device)
    return model, True


def _build_classifier(model_cfg: dict, device: torch.device) -> tuple[nn.Module, bool]:
    from src.models.resnet import CifarResNetClassifier

    model = CifarResNetClassifier(
        widen_factor=model_cfg.get("widen_factor", 10),
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    return model, False


def _build_frozen_sngp(
    model_cfg: dict, backbone_cfg: dict, device: torch.device
) -> tuple[nn.Module, bool]:
    from src.models.resnet import CifarResNetEncoder
    from src.models.frozen_sngp import FrozenBackboneSNGPClassifier

    encoder = CifarResNetEncoder(
        widen_factor=backbone_cfg.get("widen_factor", 10),
        embedding_dim=backbone_cfg["embedding_dim"],
    ).to(device)

    model = FrozenBackboneSNGPClassifier(
        encoder=encoder,
        num_classes=model_cfg["num_classes"],
        hidden_dims=model_cfg.get("hidden_dims", []),
        spec_norm_bound=model_cfg.get("spec_norm_bound", 0.95),
        dropout_rate=model_cfg.get("dropout_rate", 0.0),
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg.get("normalize_input", False),
        kernel_type=model_cfg.get("kernel_type", "normalized_rbf"),
        input_normalization=model_cfg.get("input_normalization", "l2"),
        kernel_scale=model_cfg.get("kernel_scale", 1.0),
        length_scale=model_cfg.get("length_scale", 1.0),
    ).to(device)
    return model, True


def _build_supcon_sngp(model_cfg: dict, device: torch.device) -> tuple[nn.Module, bool]:
    from src.models.supcon_sngp import CifarResNetSupConSNGPClassifier

    model = CifarResNetSupConSNGPClassifier(
        embedding_dim=model_cfg["embedding_dim"],
        num_classes=model_cfg["num_classes"],
        widen_factor=model_cfg.get("widen_factor", 10),
        hidden_dims=model_cfg.get("hidden_dims", []),
        dropout_rate=model_cfg.get("dropout_rate", 0.0),
        num_inducing=model_cfg["num_inducing"],
        ridge_penalty=model_cfg["ridge_penalty"],
        feature_scale=model_cfg["feature_scale"],
        gp_cov_momentum=model_cfg["gp_cov_momentum"],
        normalize_input=model_cfg.get("normalize_input", False),
        kernel_type=model_cfg.get("kernel_type", "normalized_rbf"),
        input_normalization=model_cfg.get("input_normalization", "l2"),
        kernel_scale=model_cfg.get("kernel_scale", 1.0),
        length_scale=model_cfg.get("length_scale", 1.0),
    ).to(device)
    return model, True


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_mc_samples: int = 10,
) -> ModelWrapper:
    """
    Loads a checkpoint and reconstructs the model.

    Reads checkpoint['config']['experiment']['name'] to detect model type:
      *_supcon_sngp       -> CifarResNetSupConSNGPClassifier
      *_frozen_sngp       -> FrozenBackboneSNGPClassifier
      *_sngp*             -> SNGPResNetClassifier  (plain sngp or sngp_augmented)
      *_classifier        -> CifarResNetClassifier

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          Target device.
        num_mc_samples:  MC samples for Laplace approximation (SNGP variants).

    Returns:
        ModelWrapper in eval mode.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    exp_name = cfg["experiment"]["name"]

    # Order matters: check more specific patterns before general ones
    if "supcon_sngp" in exp_name:
        model, has_cov = _build_supcon_sngp(model_cfg, device)
        model_type = "supcon_sngp"
    elif "frozen_sngp" in exp_name:
        backbone_cfg = cfg["backbone"]
        model, has_cov = _build_frozen_sngp(model_cfg, backbone_cfg, device)
        model_type = "frozen_sngp"
    elif "sngp" in exp_name:
        model, has_cov = _build_sngp(model_cfg, device)
        model_type = "sngp"
    elif "classifier" in exp_name:
        model, has_cov = _build_classifier(model_cfg, device)
        model_type = "classifier"
    elif "due" in exp_name:
        model, likelihood, has_cov = build_due(model_cfg, device)
        model_type = "due"
    else:
        raise ValueError(
            f"Cannot determine model type from experiment name: '{exp_name}'. "
            "Expected name to contain one of: 'supcon_sngp', 'frozen_sngp', 'sngp', 'classifier'."
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if model_type == "due":
        model.load_state_dict(state_dict["model"])
        likelihood.load_state_dict(state_dict["likelihood"])
        model.likelihood = likelihood
    else:
        model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()

    print(f"Loaded {model_type} model from '{checkpoint_path}' (exp: {exp_name})")
    return ModelWrapper(
        model=model,
        has_cov=has_cov,
        num_mc_samples=num_mc_samples,
        model_type=model_type,
        likelihood=likelihood if model_type == "due" else None,
    )
