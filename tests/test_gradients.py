"""Finite-difference vs autograd checks for K_OO and the marginal NLL."""

from __future__ import annotations

import torch

from gaussian_bn.model import GaussianDAG, k_full
from gaussian_bn.training import ObsPattern, gaussian_nll, marginal_likelihood

from _helpers import diamond_dag, gen


def _model():
    dims, edges, noise = diamond_dag(seed=5, dtype=torch.float64)
    return GaussianDAG(dims, edges, noise)


def _scalar_KOO(edge_value, m, key, observed):
    edges = dict(m.edges)
    edges[key] = edge_value
    m2 = GaussianDAG(m.dims, edges, list(m.noise), validate=False)
    oi = m2.node_index(observed)
    K_OO = k_full(m2)[oi][:, oi]
    return (K_OO ** 2).sum()  # scalar functional of K_OO


def test_fd_vs_autograd_KOO_edge():
    m = _model()
    key, observed = (1, 3), [0, 3]
    A = m.edges[key].clone().requires_grad_(True)
    f = _scalar_KOO(A, m, key, observed)
    f.backward()
    g_ad = A.grad.clone()

    eps = 1e-6
    g_fd = torch.zeros_like(A)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            e = torch.zeros_like(A); e[r, c] = eps
            fp = _scalar_KOO((A.detach() + e), m, key, observed)
            fm = _scalar_KOO((A.detach() - e), m, key, observed)
            g_fd[r, c] = (fp - fm) / (2 * eps)
    assert torch.linalg.norm(g_ad - g_fd) / torch.linalg.norm(g_fd) < 1e-6


def test_fd_vs_autograd_marginal_nll_edge():
    m = _model()
    observed = [0, 3]
    oi = m.node_index(observed)
    S = k_full(m)[oi][:, oi].detach()  # population observed cov as data
    key = (1, 3)

    def nll(Aval):
        edges = dict(m.edges); edges[key] = Aval
        m2 = GaussianDAG(m.dims, edges, list(m.noise), validate=False)
        return gaussian_nll(k_full(m2)[oi][:, oi], S)

    A = m.edges[key].clone().requires_grad_(True)
    nll(A).backward()
    g_ad = A.grad.clone()
    eps = 1e-6
    g_fd = torch.zeros_like(A)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            e = torch.zeros_like(A); e[r, c] = eps
            g_fd[r, c] = (nll(A.detach() + e) - nll(A.detach() - e)) / (2 * eps)
    # at the population optimum the gradient is ~0; perturb away first
    assert torch.linalg.norm(g_ad - g_fd) < 1e-5


def test_fd_vs_autograd_marginal_nll_noise():
    m = _model()
    observed = [0, 3]
    oi = m.node_index(observed)
    # data S from a different model so gradient is non-trivial
    g = gen(3)
    S = k_full(GaussianDAG(m.dims,
                           {k: v + 0.1 * torch.randn_like(v) for k, v in m.edges.items()},
                           list(m.noise)))[oi][:, oi].detach()

    def nll(raw):  # noise[3] = raw raw^T (PD)
        noise = list(m.noise); noise[3] = raw @ raw.mH
        m2 = GaussianDAG(m.dims, dict(m.edges), noise, validate=False)
        return gaussian_nll(k_full(m2)[oi][:, oi], S)

    L = torch.linalg.cholesky(m.noise[3]).clone().requires_grad_(True)
    nll(L).backward()
    g_ad = L.grad.clone()
    eps = 1e-6
    g_fd = torch.zeros_like(L)
    for r in range(L.shape[0]):
        for c in range(L.shape[1]):
            e = torch.zeros_like(L); e[r, c] = eps
            g_fd[r, c] = (nll(L.detach() + e) - nll(L.detach() - e)) / (2 * eps)
    assert torch.linalg.norm(g_ad - g_fd) / (torch.linalg.norm(g_fd) + 1e-12) < 1e-5


def test_nll_grad_descent_decreases():
    m = _model()
    observed = [0, 3]
    oi = m.node_index(observed)
    S = k_full(m)[oi][:, oi].detach()
    key = (1, 3)
    A = (m.edges[key] + 0.5).clone().requires_grad_(True)  # away from optimum

    def nll(Aval):
        edges = dict(m.edges); edges[key] = Aval
        m2 = GaussianDAG(m.dims, edges, list(m.noise), validate=False)
        return gaussian_nll(k_full(m2)[oi][:, oi], S)

    f0 = nll(A)
    f0.backward()
    with torch.no_grad():
        A2 = A - 0.05 * A.grad
        assert float(nll(A2)) < float(f0.detach())
