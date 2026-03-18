import torch
import gpytorch
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.models.deep_gps.dspp import DSPP
import gpytorch.settings as settings

from .gp_layers import DSPPHiddenLayer


class TwoLayerDSPPClassifier(DSPP):
    """
    Two-layer DSPP for multi-class classification.

    Dimension flow (default: latent_dim=64, hidden_dim=10, num_classes=10):
      hidden_layer : input_dims=latent_dim -> output_dims=hidden_dim  (linear mean)
      last_layer   : input_dims=hidden_dim -> output_dims=num_classes (constant mean)
      SoftmaxLikelihood -> Categorical over num_classes classes

    Typical usage:
      1. Extract features from a frozen CNN: z = cnn(x)  -> (B, latent_dim)
      2. Pass through this model: output = dspp(z)
      3. Compute loss: -objective(output, y)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        inducing_points: torch.Tensor,
        num_inducing: int,
        Q: int = 8,
    ):
        hidden_layer = DSPPHiddenLayer(
            input_dims=latent_dim,
            output_dims=hidden_dim,
            mean_type="linear",
            inducing_points=inducing_points,
            Q=Q,
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_dim,
            output_dims=num_classes,
            mean_type="constant",
            inducing_points=None,
            num_inducing=num_inducing,
            Q=Q,
        )
        likelihood = SoftmaxLikelihood(
            num_features=num_classes,
            num_classes=num_classes,
            mixing_weights=False,
        )

        super().__init__(Q)
        self.likelihood = likelihood
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

    def forward(self, inputs, **kwargs):
        hidden_rep = self.hidden_layer(inputs, **kwargs)
        return self.last_layer(hidden_rep, **kwargs)

    def predict(self, loader, cnn, device):
        """
        Run inference over a DataLoader.

        Args:
            loader: DataLoader yielding (x_batch, y_batch)
            cnn:    Feature extractor (should be in eval mode)
            device: torch.device

        Returns:
            preds     (N,)  – argmax class predictions
            log_probs (N,)  – per-sample log probabilities (nats)
        """
        cnn.eval()
        self.eval()
        all_preds, all_lls = [], []
        with settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = self(cnn(x_batch))

                base_ll = self.likelihood.log_marginal(y_batch, output)      # (Q, N)
                deep_ll = self.quad_weights.unsqueeze(-1) + base_ll           # (Q, N)
                batch_lp = deep_ll.logsumexp(dim=0)                          # (N,)
                all_lls.append(batch_lp.cpu())

                # output.mean: (Q, N, num_classes); average over Q then argmax
                all_preds.append(output.mean.mean(dim=0).argmax(dim=-1).cpu())

        return torch.cat(all_preds), torch.cat(all_lls)
