import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps.dspp import DSPP
import gpytorch.settings as settings

from .gp_layers import DSPPHiddenLayer


class TwoLayerDSPP(DSPP):
    """
    Two-layer DSPP for univariate regression.

    Dimension flow (default: input_dims=12, hidden_dim=3):
      hidden_layer : input_dims -> hidden_dim  (linear mean)
      last_layer   : hidden_dim -> None        (constant mean, scalar output)
      GaussianLikelihood

    The model output is a quadrature-weighted mixture of Q Gaussians.
    """

    def __init__(
        self,
        input_dims: int,
        inducing_points: torch.Tensor,
        num_inducing: int,
        hidden_dim: int = 3,
        Q: int = 8,
    ):
        hidden_layer = DSPPHiddenLayer(
            input_dims=input_dims,
            output_dims=hidden_dim,
            mean_type="linear",
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,  # scalar / univariate output
            mean_type="constant",
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )
        likelihood = GaussianLikelihood()

        super().__init__(Q)
        self.likelihood = likelihood
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

    def forward(self, inputs, **kwargs):
        hidden_rep = self.hidden_layer(inputs, **kwargs)
        return self.last_layer(hidden_rep, **kwargs)

    def predict(self, loader):
        """
        Run inference over a DataLoader.

        Returns:
            mus       (N,) – predictive means
            variances (N,) – predictive variances
            lls       (N,) – per-sample log probabilities (nats)
        """
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            mus, variances, lls = [], [], []
            for x_batch, y_batch in loader:
                preds = self.likelihood(self(x_batch, mean_input=x_batch))
                mus.append(preds.mean.cpu())
                variances.append(preds.variance.cpu())

                base_batch_ll = self.likelihood.log_marginal(y_batch, self(x_batch))
                deep_batch_ll = self.quad_weights.unsqueeze(-1) + base_batch_ll
                batch_log_prob = deep_batch_ll.logsumexp(dim=0)
                lls.append(batch_log_prob.cpu())

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
