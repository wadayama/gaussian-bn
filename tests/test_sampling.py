"""Tests for gaussian_bn.inference.sample: shape, reproducibility, MC convergence."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import sample
from gaussian_bn.model import GaussianDAG, k_full

from _helpers import diamond_dag, gen, relerr_fro


def _model(builder, *, seed, dtype, **kw):
    dims, edges, noise = builder(seed=seed, dtype=dtype, **kw)
    return GaussianDAG(dims, edges, noise, dtype=dtype)


def test_sample_shape_dtype():
    m = _model(diamond_dag, seed=1, dtype=torch.float64)
    X = sample(m, 16, gen(0))
    assert X.shape == (16, m.D)
    assert X.dtype == torch.float64


def test_sample_reproducible():
    m = _model(diamond_dag, seed=1, dtype=torch.float64)
    X1 = sample(m, 100, gen(7))
    X2 = sample(m, 100, gen(7))
    assert torch.equal(X1, X2)


def test_sample_cov_converges_diamond():
    m = _model(diamond_dag, seed=2, dtype=torch.float64)
    Ktrue = k_full(m)
    errs = []
    for N in [1000, 10000, 100000]:
        X = sample(m, N, gen(100 + N % 7))
        S = (X.mH @ X) / N
        errs.append(relerr_fro(S, Ktrue))
    assert errs[-1] < 0.02


@pytest.mark.slow
def test_sample_cov_convergence_rate():
    m = _model(diamond_dag, seed=2, dtype=torch.float64)
    Ktrue = k_full(m)
    e_lo = relerr_fro((lambda X: (X.mH @ X) / X.shape[0])(sample(m, 1000, gen(1))), Ktrue)
    e_hi = relerr_fro((lambda X: (X.mH @ X) / X.shape[0])(sample(m, 1_000_000, gen(2))), Ktrue)
    ratio = e_lo / e_hi
    assert 0.5 * (1000_000 / 1000) ** 0.5 < ratio < 2.0 * (1000_000 / 1000) ** 0.5


@pytest.mark.slow
def test_sample_cov_complex():
    m = _model(diamond_dag, seed=3, dtype=torch.complex128)
    Ktrue = k_full(m)
    X = sample(m, 200000, gen(5))
    S = (X.mH @ X) / X.shape[0]
    assert relerr_fro(S, Ktrue) < 0.03
