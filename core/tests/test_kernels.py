from core.kernels import SinkhornDistance, emd_distance, EmdRbfKernel
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import gpytorch.settings as gpytorch_settings


def test_sinkhorn_distance():
    n = 5
    x = torch.tensor([[0, i] for i in range(n)], dtype=torch.float)
    y = torch.tensor([[1, i] for i in range(n)], dtype=torch.float)

    distance_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = x[i] - y[j]
            distance_matrix[i, j] = (diff[0] ** 2 + diff[1] ** 2)

    distance_metric = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    dist, coupling_matrix, results = distance_metric(x, y)
    assert torch.allclose(dist, torch.tensor(1, dtype=torch.float), atol=1e-4)
    assert torch.allclose(results, distance_matrix, atol=1e-4)


def test_emd_distance():
    n = 5
    x = torch.tensor([[0, i] for i in range(n)], dtype=torch.float)
    y = torch.tensor([[1, i] for i in range(n)], dtype=torch.float)

    distance_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = x[i] - y[j]
            distance_matrix[i, j] = (diff[0] ** 2 + diff[1] ** 2)

    assert torch.allclose(emd_distance(x, y, x1_eq_x2=False), distance_matrix, atol=1e-4)


# TODO: assertions
def test_emd_rbf_kernel():
    n = 3
    x = torch.tensor([[0, i] for i in range(n)], dtype=torch.float)

    # Non-batch: Simple option
    covar_module = EmdRbfKernel
    # Non-batch: ARD (different lengthscale for each input dimension)
    covar_module = EmdRbfKernel()
    covar = covar_module(x)  # Output: LinearOperator of size (10 x 10)

    distance_matrix = torch.tensor([[0., 1., 4.], [1., 0., 1.], [4., 1., 0.]])

    assert torch.allclose(covar_module.covar_dist(x, x), distance_matrix, atol=1e-4)

    covar.to_dense()

    batch_x = torch.randn(2, 10, 5)
    # Batch: Simple option
    covar_module = EmdRbfKernel()
    # Batch: different lengthscale for each batch
    covar_module = EmdRbfKernel(batch_shape=torch.Size([2]))
    covar = covar_module(x)  # Output: LinearOperator of size (2 x 10 x 10)
    pass


# TODO: assertions
def test_emd_gp():
    n = 3
    train_X = torch.tensor([[0, i] for i in range(n)], dtype=torch.float)
    train_Y = torch.sin(train_X[:, -1]).unsqueeze(-1)

    surrogate = SingleTaskGP(train_X, train_Y, covar_module=EmdRbfKernel())
    likelihood = surrogate.likelihood

    surrogate.train()
    likelihood.train()

    mll = ExactMarginalLogLikelihood(likelihood, surrogate)
    mll = fit_gpytorch_mll(mll)

    surrogate.eval()
    likelihood.eval()

    test_x = torch.tensor([[0, i] for i in range(n+2)], dtype=torch.float)

    with torch.no_grad(), gpytorch_settings.fast_pred_var():
        prediction = likelihood(surrogate(test_x))

    lower, upper = prediction.confidence_region()
    mean = prediction.mean

    pass
