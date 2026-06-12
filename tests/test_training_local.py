"""Tests for full-observation local-regression MLE (with ridge)."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import sample
from gaussian_bn.model import GaussianDAG, k_full
from gaussian_bn.training import fit_local_regression, local_mle_from_cov

from _helpers import ALL_DTYPES, ATOL_EXACT, diamond_dag, gen


def _model(builder, *, seed, dtype, **kw):
    dims, edges, noise = builder(seed=seed, dtype=dtype, **kw)
    return GaussianDAG(dims, edges, noise, dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_local_mle_exact_from_population_cov(dtype):
    # population covariance -> MLE is exact at the true parameters
    m = _model(diamond_dag, seed=2, dtype=dtype)
    Ktrue = k_full(m)
    eh, nh = local_mle_from_cov(m, Ktrue)
    for key in m.edges:
        assert torch.allclose(eh[key], m.edges[key], atol=ATOL_EXACT)
    for j in range(m.M):
        assert torch.allclose(nh[j], m.noise[j], atol=ATOL_EXACT)


def test_fit_local_regression_recovery():
    m = _model(diamond_dag, seed=3, dtype=torch.float64)
    errs = []
    for N in [2000, 20000, 200000]:
        X = sample(m, N, gen(7 + N % 5))
        eh, nh = fit_local_regression(m, X)
        a_err = max(float(torch.linalg.norm(eh[k] - m.edges[k])) for k in m.edges)
        s_err = max(float(torch.linalg.norm(nh[j] - m.noise[j])) for j in range(m.M))
        errs.append((a_err, s_err))
    assert errs[-1][0] < 0.05 and errs[-1][1] < 0.05


def test_local_mle_root_noise():
    m = _model(diamond_dag, seed=4, dtype=torch.float64)
    Ktrue = k_full(m)
    eh, nh = local_mle_from_cov(m, Ktrue)
    # root node 0: estimate equals its own covariance block
    assert torch.allclose(nh[0], m.noise[0], atol=ATOL_EXACT)


def test_ridge_matches_closed_form_and_shrinks():
    m = _model(diamond_dag, seed=5, dtype=torch.float64)
    X = sample(m, 5000, gen(11))
    N = X.shape[0]
    Kmat = (X.mH @ X) / N
    lam = 0.5
    eh0, _ = local_mle_from_cov(m, Kmat, ridge=0.0)
    ehr, _ = local_mle_from_cov(m, Kmat, ridge=lam)
    # closed-form ridge for node 3 (parents 1,2)
    j = 3
    sj = m.node_index([j]); ui = m.node_index(m.parents[j])
    KjU = Kmat[sj][:, ui]; KUU = Kmat[ui][:, ui]
    B_ref = KjU @ torch.linalg.inv(KUU + lam * torch.eye(KUU.shape[0], dtype=torch.float64))
    B_got = torch.cat([ehr[(i, j)] for i in m.parents[j]], dim=1)
    assert torch.allclose(B_got, B_ref, atol=1e-8)
    # ridge shrinks the edge norms relative to OLS
    n0 = sum(float(torch.linalg.norm(eh0[k])) for k in m.edges)
    nr = sum(float(torch.linalg.norm(ehr[k])) for k in m.edges)
    assert nr < n0
