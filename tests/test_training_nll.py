"""Tests for gaussian_nll, marginal_likelihood (multi-pattern), and fit_gradient."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import sample
from gaussian_bn.model import GaussianDAG, k_full
from gaussian_bn.training import (
    ObsPattern,
    fit_gradient,
    fit_gradient_custom,
    gaussian_nll,
    marginal_likelihood,
)

from _helpers import ATOL_EXACT, gen, relerr_fro, spd


def hidden_diamond():
    dims = [1, 1, 1, 1]
    edges = {(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]}
    noise = [[[1.0]], [[0.4]], [[0.5]], [[0.3]]]
    return GaussianDAG(dims, edges, noise)


def test_gaussian_nll_value():
    K = spd(3, seed=1, dtype=torch.float64)
    S = spd(3, seed=2, dtype=torch.float64)
    ref = torch.linalg.slogdet(K).logabsdet + torch.trace(torch.linalg.inv(K) @ S)
    assert abs(float(gaussian_nll(K, S) - ref)) < ATOL_EXACT


def test_gaussian_nll_minimized_at_S():
    S = spd(3, seed=3, dtype=torch.float64)
    base = float(gaussian_nll(S, S))
    # value at optimum equals logdet S + dim
    assert abs(base - float(torch.linalg.slogdet(S).logabsdet + 3)) < 1e-9
    # perturbing K away from S increases the NLL
    P = spd(3, seed=4, dtype=torch.float64)
    worse = float(gaussian_nll(S + 0.3 * P, S))
    assert worse > base


def test_gaussian_nll_solve_vs_inv():
    K = spd(4, seed=5, dtype=torch.float64)
    S = spd(4, seed=6, dtype=torch.float64)
    ref = torch.linalg.slogdet(K).logabsdet + torch.trace(torch.linalg.inv(K) @ S)
    assert abs(float(gaussian_nll(K, S) - ref)) < ATOL_EXACT


def test_fit_gradient_recovers_observed_cov_population():
    # use the population observed covariance as "data" S -> exact optimum K_OO = S
    m = hidden_diamond()
    observed = [0, 3]
    oi = m.node_index(observed)
    K_OO_true = k_full(m)[oi][:, oi]
    # build a single ObsPattern whose sample-cov equals K_OO_true by using a
    # whitening "dataset": Y = chol(K_OO_true)^T so (Y^T Y)/N == K_OO_true with N=dim
    L = torch.linalg.cholesky(K_OO_true)
    Y = L.mH * (2 ** 0.5)  # 2 rows so (Y^H Y)/2 = K_OO_true? construct exactly below
    # simplest exact construction: N synthetic rows with empirical cov = K_OO_true
    # use eigendecomposition: rows = sqrt(N) * eigvec scaled
    w, V = torch.linalg.eigh(K_OO_true)
    N = K_OO_true.shape[0]
    Y = (V @ torch.diag(torch.sqrt(w * N))).mH  # (N, dim), (Y^H Y)/N == K_OO_true
    S = (Y.mH @ Y) / N
    assert torch.allclose(S, K_OO_true, atol=1e-10)

    # random init model (same structure), fit by LBFGS
    g = gen(21)
    init_edges = {k: torch.randn(1, 1, generator=g, dtype=torch.float64) for k in m.edges}
    init_noise = [torch.tensor([[0.7]], dtype=torch.float64) for _ in range(4)]
    m0 = GaussianDAG(m.dims, init_edges, init_noise)
    fitted, hist = fit_gradient(m0, [ObsPattern(tuple(observed), Y)],
                                optimizer="lbfgs", lr=1.0, num_iters=500)
    K_OO_hat = k_full(fitted)[oi][:, oi]
    assert relerr_fro(K_OO_hat, S) < 1e-6
    assert hist[-1] < hist[0]


def test_marginal_likelihood_multipattern():
    m = hidden_diamond()
    # two patterns: observe {0,3} and {0,2,3}
    X = sample(m, 2000, gen(31))
    p1 = ObsPattern((0, 3), X[:1000][:, m.node_index([0, 3])])
    p2 = ObsPattern((0, 2, 3), X[1000:][:, m.node_index([0, 2, 3])])
    total = marginal_likelihood(m, [p1, p2])
    # hand-sum: n1*nll1_perrow_mean style -> recompute the weighted average
    from gaussian_bn.linalg import logdet_hpd, solve_psd
    Kf = k_full(m)

    def patt(pat):
        oi = m.node_index(pat.observed)
        K_OO = Kf[oi][:, oi]
        Y = pat.Y
        ld = logdet_hpd(K_OO)
        quad = (Y.conj() * solve_psd(K_OO, Y.mH).mH).real.sum()
        return Y.shape[0] * ld + quad, Y.shape[0]

    s1, n1 = patt(p1)
    s2, n2 = patt(p2)
    ref = (s1 + s2) / (n1 + n2)
    assert abs(float(total - ref)) < 1e-9


def test_fit_gradient_custom_shared_factor():
    # edges 0->1 = H1 F and 0->2 = H2 F share the factor F; learn F from data.
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(10)
    H1 = torch.randn(2, 2, generator=g)
    H2 = torch.randn(2, 2, generator=g)
    F_true = torch.tensor([[0.6, -0.3], [0.2, 0.5]])
    true = GaussianDAG([2, 2, 2], {(0, 1): H1 @ F_true, (0, 2): H2 @ F_true},
                       [torch.eye(2), 0.3 * torch.eye(2), 0.3 * torch.eye(2)])
    X = sample(true, 50_000, gen(11))

    F = (0.1 * torch.randn(2, 2, generator=gen(12))).requires_grad_(True)
    s = [torch.zeros(2, requires_grad=True) for _ in range(3)]

    def build():
        return GaussianDAG([2, 2, 2], {(0, 1): H1 @ F, (0, 2): H2 @ F},
                           [torch.diag(torch.exp(si)) for si in s], validate=False)

    hist = fit_gradient_custom([F, *s], build, [ObsPattern((0, 1, 2), X)],
                          optimizer="adam", lr=0.02, num_iters=1500)
    assert hist[-1] < hist[0]
    assert float(torch.linalg.norm(F.detach() - F_true)) < 0.05


def test_fit_gradient_custom_rejects_non_grad_params():
    m = hidden_diamond()
    with pytest.raises(ValueError):
        fit_gradient_custom([torch.zeros(2)], lambda: m,
                       [ObsPattern((0, 3), torch.zeros(1, 2))], num_iters=1)


def test_gaussian_nll_grad_nonzero():
    m = hidden_diamond()
    observed = [0, 3]
    X = sample(m, 1000, gen(41))
    Y = X[:, m.node_index(observed)]
    A = m.edges[(1, 3)].clone().requires_grad_(True)
    edges = dict(m.edges); edges[(1, 3)] = A
    m2 = GaussianDAG(m.dims, edges, list(m.noise), validate=False)
    oi = m2.node_index(observed)
    S = (Y.mH @ Y) / Y.shape[0]
    loss = gaussian_nll(k_full(m2)[oi][:, oi], S)
    loss.backward()
    assert A.grad is not None and A.grad.abs().sum() > 1e-6
