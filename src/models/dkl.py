import torch

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)


class GP(ApproximateGP):
    # TODO: Add support for different kernels
    def __init__(self, inducing_points, num_inducing, num_output=1, per_feature=False):
        
        if per_feature:
            # Learn a separate GP for each feature (e.g. for DKL)
            inducing_points = inducing_points.transpose(-1, -2).unsqueeze(-1)
            batch_shape = torch.Size([num_output])
        elif num_output > 1:
            batch_shape = torch.Size([num_output])
        else:
            batch_shape = torch.Size([])

            
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        if num_output > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_output
            )
        
        super(GP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, gp_layer, likelihood, per_feature=False):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer
        self.likelihood = likelihood
        self.per_feature = per_feature

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.per_feature:
            features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
    
    def predict(self, loader, cnn=None, device=None):
        """
        Run inference over a DataLoader for DKL model.

        Args:
            loader: DataLoader yielding (x_batch, y_batch)
            cnn:    Ignored for DKL (feature extractor is built into the model)
            device: torch.device

        Returns:
            preds     (N,)  – argmax class predictions
            log_probs (N,)  – per-sample log probabilities (nats)
        """
        self.eval()
        self.likelihood.eval()
        all_preds, all_lls = [], []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass through the entire model (feature extraction + GP)
                output = self(x_batch)
                
                # Get log marginal from the GP layer's likelihood
                # Need access to the GP's likelihood/objective from outside
                base_ll = self.likelihood.log_marginal(y_batch, output)
                all_lls.append(base_ll.cpu())
                
                # Get predictions from mean of distribution
                all_preds.append(self.likelihood(output).probs.mean(0).argmax(dim=-1).cpu())

        return torch.cat(all_preds), torch.cat(all_lls)