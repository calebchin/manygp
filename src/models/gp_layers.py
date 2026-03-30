import torch
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution, IndependentMultitaskVariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.models.deep_gps.dspp import DSPPLayer

from gpytorch.variational import VariationalStrategy

class DSPPHiddenLayer(DSPPLayer):
    """
    A single hidden (or output) layer for a Deep Semi-Parametric GP (DSPP).

    Uses a Matern kernel with ARD lengthscales and mean-field variational inference.
    Supports 'constant' mean (for output layers) and 'linear' mean (for hidden layers,
    following Salimbeni et al. 2017).
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        num_inducing=300,
        inducing_points=None,
        mean_type="constant",
        Q=8,
    ):
        if inducing_points is not None and output_dims is not None and inducing_points.dim() == 2:
            # Expand 2D inducing points to match the number of GPs in this layer.
            inducing_points = inducing_points.unsqueeze(0).expand(
                (output_dims,) + inducing_points.shape
            )
            inducing_points = inducing_points.clone() + 0.01 * torch.randn_like(inducing_points)

        if inducing_points is None:
            if output_dims is None:
                # Single GP (e.g. univariate regression output layer)
                inducing_points = torch.randn(num_inducing, input_dims)
            else:
                inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        else:
            num_inducing = inducing_points.size(-2)

        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([]),
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        batch_shape = torch.Size([]) if output_dims is None else torch.Size([output_dims])
        super().__init__(variational_strategy, input_dims, output_dims, Q)

        if mean_type == "constant":
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        elif mean_type == "linear":
            self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        else:
            raise ValueError(f"Unknown mean_type: {mean_type!r}. Use 'constant' or 'linear'.")

        self.covar_module = ScaleKernel(
            MaternKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x, mean_input=None, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
