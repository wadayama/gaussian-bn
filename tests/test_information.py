"""Tests for gaussian_bn.inference: marginal, conditioning, MI, CMI, logdet_hpd."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import (
    conditional_covariance,
    conditional_mean,
    conditional_mutual_information,
    marginal,
    mutual_information,
    sample,
)
from gaussian_bn.linalg import logdet_hpd
from gaussian_bn.model import GaussianDAG, k_full

from _helpers import (
    ALL_DTYPES,
    ATOL_EXACT,
    ATOL_TIGHT,
    chain_dag,
    collider_dag,
    diamond_dag,
    empirical_regression,
    gen,
    is_hermitian,
    is_psd,
    precision_conditional,
    spd,
)


def _model(builder, *, seed, dtype, **kw):
    dims, edges, noise = builder(seed=seed, dtype=dtype, **kw)
    return GaussianDAG(dims, edges, noise, dtype=dtype)


# --------------------------------------------------------------------------
# logdet_hpd
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("d", [1, 2, 3, 5])
def test_logdet_hpd_identity(dtype, d):
    assert abs(float(logdet_hpd(torch.eye(d, dtype=dtype)))) < ATOL_TIGHT


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_logdet_hpd_vs_slogdet(dtype):
    S = spd(4, seed=1, dtype=dtype)
    ref = torch.linalg.slogdet(S).logabsdet
    assert abs(float(logdet_hpd(S) - ref)) < ATOL_EXACT


def test_logdet_hpd_jitter_illconditioned():
    # eigenvalues ~ [1, 1e-14]; jitter floors the determinant
    Q = torch.linalg.qr(torch.randn(2, 2, dtype=torch.float64))[0]
    S = Q @ torch.diag(torch.tensor([1.0, 1e-14], dtype=torch.float64)) @ Q.T
    val = logdet_hpd(S, jitter=1e-8)
    ref = torch.log(torch.tensor(1.0 + 1e-8) * torch.tensor(1e-14 + 1e-8))
    assert torch.isfinite(val)
    assert abs(float(val - ref)) < 1e-3


# --------------------------------------------------------------------------
# conditional independence via CMI (chain + collider)
# --------------------------------------------------------------------------
def test_chain_cmi_conditional_independence():
    m = GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                    [[[1.0]], [[0.5]], [[0.5]]])
    Kf = k_full(m)
    assert float(mutual_information(m, [0], [2], Kf)) > 1e-3
    assert abs(float(conditional_mutual_information(m, [0], [2], [1], Kf))) < 1e-10


def test_collider_cmi_conditioning_creates_dependence():
    m = GaussianDAG([1, 1, 1], {(0, 2): [[1.1]], (1, 2): [[0.7]]},
                    [[[1.0]], [[1.0]], [[0.3]]])
    Kf = k_full(m)
    assert abs(float(mutual_information(m, [0], [1], Kf))) < 1e-10
    assert float(conditional_mutual_information(m, [0], [1], [2], Kf)) > 1e-3


# --------------------------------------------------------------------------
# marginal / conditioning correctness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_marginal_is_submatrix(dtype):
    m = _model(diamond_dag, seed=2, dtype=dtype)
    Kf = k_full(m)
    idx = m.node_index([1, 3])
    assert torch.allclose(marginal(m, [1, 3], Kf), Kf[idx][:, idx], atol=ATOL_TIGHT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_conditional_cov_schur_vs_precision(dtype):
    m = _model(diamond_dag, seed=3, dtype=dtype)
    Kf = k_full(m)
    A, B = [3], [0, 1]
    C_schur = conditional_covariance(m, A, B, Kf)
    M_prec, C_prec = precision_conditional(Kf, m.node_index(A), m.node_index(B))
    M_schur = marginal_cross(m, Kf, A, B) @ torch.linalg.inv(marginal(m, B, Kf))
    assert torch.allclose(C_schur, C_prec, atol=ATOL_EXACT)
    assert torch.allclose(M_schur, M_prec, atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_conditional_mean_closure(dtype):
    m = _model(diamond_dag, seed=4, dtype=dtype)
    Kf = k_full(m)
    A, B = [3], [0, 1]
    b = torch.arange(1, m.node_index(B).numel() + 1, dtype=torch.float64).to(dtype)
    M = marginal_cross(m, Kf, A, B) @ torch.linalg.inv(marginal(m, B, Kf))
    assert torch.allclose(conditional_mean(m, A, B, b, Kf), M @ b, atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_conditional_cov_solve_vs_inv(dtype):
    m = _model(diamond_dag, seed=5, dtype=dtype)
    Kf = k_full(m)
    A, B = [2], [0, 3]
    KAA = marginal(m, A, Kf)
    KAB = marginal_cross(m, Kf, A, B)
    KBB = marginal(m, B, Kf)
    explicit = KAA - KAB @ torch.linalg.inv(KBB) @ KAB.mH
    assert torch.allclose(conditional_covariance(m, A, B, Kf), explicit, atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_conditional_covs_hermitian_psd(dtype):
    m = _model(diamond_dag, seed=6, dtype=dtype)
    Kf = k_full(m)
    for A, B in [([3], [0]), ([2, 3], [0, 1]), ([0], [1, 2, 3])]:
        C = conditional_covariance(m, A, B, Kf)
        assert is_hermitian(C, atol=ATOL_EXACT)
        assert is_psd(C)


def test_conditional_cov_matches_empirical():
    m = _model(diamond_dag, seed=7, dtype=torch.float64)
    Kf = k_full(m)
    A, B = [3], [0, 1]
    X = sample(m, 400_000, gen(99))
    VA = X[:, m.slc(3)]
    VB = torch.cat([X[:, m.slc(0)], X[:, m.slc(1)]], dim=1)
    M_emp, C_emp = empirical_regression(VA, VB)
    C_schur = conditional_covariance(m, A, B, Kf)
    assert torch.linalg.norm(C_schur - C_emp) / torch.linalg.norm(C_schur) < 1e-2


# helper local to this file: cross block K_AB
def marginal_cross(model, Kf, A, B):
    ri = model.node_index(A)
    ci = model.node_index(B)
    return Kf[ri][:, ci]
