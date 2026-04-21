"""
Unified lightweight model wrapper for post-training evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelWrapper:
    """Unified inference interface for classifiers and SNGP variants."""

    def __init__(
        self,
        model: nn.Module,
        has_cov: bool,
        num_mc_samples: int = 10,
        model_type: str = "",
    ):
        self.model = model
        self.has_cov = has_cov
        self.num_mc_samples = num_mc_samples
        self.model_type = model_type

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (logits, variances). variances is None for plain classifiers."""
        if self.has_cov:
            logits, variances = self.model(x, return_cov=True)
            return logits, variances
        logits = self.model(x)
        return logits, None

    @torch.no_grad()
    def predict_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictive probabilities, using Laplace for SNGP models."""
        from src.models.sngp import laplace_predictive_probs

        logits, variances = self.predict_logits(x)
        if self.has_cov and variances is not None:
            return laplace_predictive_probs(logits, variances, num_mc_samples=self.num_mc_samples)
        return F.softmax(logits, dim=-1)