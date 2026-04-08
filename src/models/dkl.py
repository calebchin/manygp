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

from sklearn import cluster

def initial_values(train_dataset, feature_extractor, n_inducing_points):
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample = torch.stack([train_dataset[j][0] for j in idx[i]])

            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()


class GP(ApproximateGP):
    # TODO: Add support for different kernels
    def __init__(
        self,
        num_outputs,
        initial_lengthscale,
        initial_inducing_points,
        kernel="RBF",
		per_feature=False
    ):
        n_inducing_points = initial_inducing_points.shape[0]

        if per_feature:
            # Learn a separate GP for each feature (e.g. for DKL)
            inducing_points = inducing_points.transpose(-1, -2).unsqueeze(-1)
            batch_shape = torch.Size([num_outputs])
        elif num_outputs > 1:
            batch_shape = torch.Size([num_outputs])
        else:
            batch_shape = torch.Size([])

            
        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points, batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self, initial_inducing_points, variational_distribution
        )

        if num_outputs > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_outputs
            )
        
        super(GP, self).__init__(variational_strategy)
        kwargs = {
            "batch_shape": batch_shape,
        }

        if kernel == "RBF":
            kernel = RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def inducing_points(self):
        for name, param in self.named_parameters():
            if "inducing_points" in name:
                return param


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
