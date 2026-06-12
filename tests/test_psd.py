"""Tests for PSD (singular-noise / deterministic-node) support.

Covers the ``validate="psd"`` constructor mode, the ``psd_factor`` fallback,
sampling through deterministic nodes, and the exactness of the zero-noise
chain against the collapsed "star" representation -- plus the backward
compatibility guarantee that PD models keep using the Cholesky path.
"""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import marginal, sample
from gaussian_bn.linalg import cholesky_psd, psd_factor
from gaussian_bn.model import GaussianDAG

from _helpers import gen, relerr_fro


def _rotation(theta: float, rho: float = 0.95) -> torch.Tensor:
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return rho * torch.tensor([[c, -s], [s, c]], dtype=torch.float64)


def _ode_chain(T: int):
    """Deterministic d=2 Euler chain x_{t+1} = F x_t with noisy scalar obs."""
    F = _rotation(0.5)
    H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64)
    R = torch.tensor([[0.1]], dtype=torch.float64)
    dims = [2] * (T + 1) + [1] * (T + 1)
    edges = {(t, t + 1): F for t in range(T)}
    edges.update({(t, T + 1 + t): H for t in range(T + 1)})
    noise = [P0] + [torch.zeros(2, 2, dtype=torch.float64)] * T \
        + [R.clone() for _ in range(T + 1)]
    m = GaussianDAG(dims, edges, noise, validate="psd")
    return m, F, H, P0, R


def test_psd_mode_accepts_zero_noise_default_rejects():
    dims = [1, 1]
    edges = {(0, 1): [[1.0]]}
    noise = [[[1.0]], [[0.0]]]
    with pytest.raises(ValueError, match="positive definite"):
        GaussianDAG(dims, edges, noise)                      # default: PD required
    m = GaussianDAG(dims, edges, noise, validate="psd")      # PSD: accepted
    assert m.M == 2


def test_psd_mode_rejects_indefinite():
    with pytest.raises(ValueError, match="semi-definite"):
        GaussianDAG([1, 1], {(0, 1): [[1.0]]}, [[[1.0]], [[-0.5]]], validate="psd")


def test_psd_factor_matches_cholesky_on_pd():
    # Backward compatibility: PD input takes the Cholesky path bit-for-bit.
    A = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    assert torch.equal(psd_factor(A), cholesky_psd(A))


def test_psd_factor_zero_and_singular():
    Z = torch.zeros(2, 2, dtype=torch.float64)
    assert torch.equal(psd_factor(Z), Z)                     # L L^H = 0
    v = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    A = v @ v.mT                                             # rank-1 PSD
    L = psd_factor(A)
    assert relerr_fro(L @ L.mH, A) < 1e-12


def test_zero_noise_chain_equals_star_representation():
    T = 5
    m, F, H, P0, R = _ode_chain(T)
    y_nodes = list(range(T + 1, 2 * T + 2))
    K_chain = marginal(m, y_nodes)

    edges_s, Ft = {}, torch.eye(2, dtype=torch.float64)
    for t in range(T + 1):
        edges_s[(0, 1 + t)] = H @ Ft
        Ft = F @ Ft
    m_star = GaussianDAG([2] + [1] * (T + 1), edges_s,
                         [P0] + [R.clone() for _ in range(T + 1)])
    K_star = marginal(m_star, list(range(1, T + 2)))
    assert torch.allclose(K_chain, K_star, atol=1e-14)


def test_sample_through_deterministic_nodes():
    T = 4
    m, F, _, _, _ = _ode_chain(T)
    X = sample(m, 2000, gen(11))
    # hidden states obey the deterministic update exactly
    for t in range(T):
        xt = X[:, m.slc(t)]
        xt1 = X[:, m.slc(t + 1)]
        assert torch.allclose(xt1, xt @ F.mT, atol=1e-12)
    # observed covariance matches the model covariance statistically
    y_nodes = list(range(T + 1, 2 * T + 2))
    yi = m.node_index(y_nodes)
    S = (X[:, yi].mH @ X[:, yi]) / X.shape[0]
    assert relerr_fro(S, marginal(m, y_nodes)) < 0.1


def test_validate_mode_survives_functional_updates():
    m, _, _, _, R = _ode_chain(3)
    m2 = m.replace_noise(2 * 3 + 1, 2.0 * R)     # touch an observation node
    assert m2.validate == "psd"
    m3 = m2.replace_edges({(0, 1): torch.eye(2, dtype=torch.float64)})
    assert m3.validate == "psd"                   # zero noise still accepted
