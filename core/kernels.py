from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
import torch
from torch import Tensor

import torch.nn as nn


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    :eps: regularization coefficient
    :max_iter: maximum number of Sinkhorn iterations
    :reduction: Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'none'
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        """
        TODO: reformat doc to gpytorch standards
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
        """
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def emd_distance(x1: Tensor, x2: Tensor, x1_eq_x2: bool) -> Tensor:
    if x1_eq_x2:
        # TODO: implement case for x1 is fully equal to x2
        pass
    # distance, _, _ = sinkhorn(x1, x2, p=2)
    distance_metric = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    dist, coupling_matrix, distance_matrix = distance_metric(x1, x2)
    return distance_matrix


class EmdRbfKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Earth Mover's Distance RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel` from the gpytorch library.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)
    :param eps: The minimum value that the lengthscale can take (prevents
        divide by zero errors). (Default: `1e-6`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
    """

    has_lengthscale = True

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Earth Mover's Distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        if diag:
            raise NotImplementedError
        else:
            dist_func = emd_distance
            return dist_func(x1, x2, x1_eq_x2)

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return postprocess_rbf(self.covar_dist(x1_, x2_, diag=diag, **params))
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            # TODO: square distance
            lambda x1, x2: self.covar_dist(x1, x2, diag=False, **params),
        )
