"""Tests for EM training: posterior moments, monotone NLL, observed-cov recovery."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import sample
from gaussian_bn.model import GaussianDAG, k_full
from gaussian_bn.training import em_fit, posterior_full_moments

from _helpers import gen, is_hermitian, is_psd, relerr_fro


def hidden_diamond():
    dims = [1, 1, 1, 1]
    edges = {(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]}
    noise = [[[1.0]], [[0.4]], [[0.5]], [[0.3]]]
    return GaussianDAG(dims, edges, noise)


def test_posterior_moments_fully_observed_equals_sample_cov():
    m = hidden_diamond()
    X = sample(m, 5000, gen(1))
    all_nodes = list(range(m.M))
    Ktil = posterior_full_moments(m, all_nodes, X)
    S = (X.mH @ X) / X.shape[0]
    assert torch.allclose(Ktil, S, atol=1e-8)


def test_posterior_moments_hermitian_psd():
    m = hidden_diamond()
    X = sample(m, 5000, gen(2))
    Ktil = posterior_full_moments(m, [0, 3], X[:, m.node_index([0, 3])])
    assert is_hermitian(Ktil, atol=1e-10)
    assert is_psd(Ktil)


def test_em_monotone_and_recovers_observed_cov():
    m = hidden_diamond()
    observed = [0, 3]
    oi = m.node_index(observed)
    X = sample(m, 200000, gen(30))
    S = (X[:, oi].mH @ X[:, oi]) / X.shape[0]

    g = gen(31)
    init_edges = {k: torch.randn(1, 1, generator=g, dtype=torch.float64) for k in m.edges}
    init_noise = [torch.tensor([[0.7]], dtype=torch.float64) for _ in range(4)]
    m0 = GaussianDAG(m.dims, init_edges, init_noise)
    fitted, hist = em_fit(m0, X, observed, num_iters=60)

    diffs = [hist[t + 1] - hist[t] for t in range(len(hist) - 1)]
    assert max(diffs) < 1e-6                      # monotone (within rounding)
    K_OO_hat = k_full(fitted)[oi][:, oi]
    assert relerr_fro(K_OO_hat, S) < 1e-3


def test_em_rejects_affine_model():
    # The EM path is zero-mean only: an affine model must be rejected loudly,
    # not silently fitted with the mean absorbed into the covariances.
    dims = [1, 1, 1]
    edges = {(0, 1): [[0.8]], (1, 2): [[0.9]]}
    noise = [[[1.0]], [[0.5]], [[0.4]]]
    mean = [torch.tensor([5.0]), torch.tensor([-3.0]), torch.tensor([2.0])]
    m = GaussianDAG(dims, edges, noise, mean=mean)
    X = sample(m, 100, gen(50))
    with pytest.raises(ValueError, match="zero-mean"):
        em_fit(m, X, observed=[0, 2], num_iters=3)
    with pytest.raises(ValueError, match="zero-mean"):
        posterior_full_moments(m, [0, 2], X[:, m.node_index([0, 2])])


def test_em_population_fixed_point():
    # initialize EM at the true parameters; observed cov stays put
    m = hidden_diamond()
    observed = [0, 3]
    oi = m.node_index(observed)
    X = sample(m, 200000, gen(40))
    fitted, hist = em_fit(m, X, observed, num_iters=5)
    K_OO_start = k_full(m)[oi][:, oi]
    K_OO_end = k_full(fitted)[oi][:, oi]
    assert relerr_fro(K_OO_end, K_OO_start) < 1e-2
