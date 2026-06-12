"""Property-based invariant tests holding for arbitrary random parameters.

Deterministic loops over seeds (no Hypothesis) to match the reference style.
"""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import (
    conditional_covariance,
    conditional_mutual_information,
    marginal,
    mutual_information,
)
from gaussian_bn.model import GaussianDAG, k_full

from _helpers import ALL_DTYPES, ATOL_INVARIANT, ATOL_TIGHT, PSD_TOL, chain_dag, random_dag


def _model(builder, *, seed, dtype, **kw):
    dims, edges, noise = builder(seed=seed, dtype=dtype, **kw)
    return GaussianDAG(dims, edges, noise, dtype=dtype)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_marginalization_consistency(dtype):
    m = _model(random_dag, seed=1, dtype=dtype, M=5)
    Kf = k_full(m)
    sub = marginal(m, [0, 2, 4], Kf)
    # marginal over {0,4} of the full equals marginal over {0,4} of the {0,2,4} block
    idx_full = m.node_index([0, 4])
    direct = Kf[idx_full][:, idx_full]
    # build index into the sub-block
    d0 = m.dims[0]
    d2 = m.dims[2]
    keep = list(range(d0)) + list(range(d0 + d2, d0 + d2 + m.dims[4]))
    keep = torch.tensor(keep)
    nested = sub[keep][:, keep]
    assert torch.allclose(direct, nested, atol=ATOL_TIGHT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_mi_cmi_symmetry(dtype):
    for seed in range(6):
        m = _model(random_dag, seed=seed, dtype=dtype, M=5)
        Kf = k_full(m)
        A, B, C = [0], [4], [2]
        assert abs(float(mutual_information(m, A, B, Kf)
                         - mutual_information(m, B, A, Kf))) < ATOL_INVARIANT
        assert abs(float(conditional_mutual_information(m, A, B, C, Kf)
                         - conditional_mutual_information(m, B, A, C, Kf))) < ATOL_INVARIANT


def test_cmi_zero_under_dsep_chain():
    # chain 0->1->2->3: 0 _|_ 3 | {1,2}; 0 _|_ 2 | 1
    for seed in range(6):
        m = _model(chain_dag, seed=seed, dtype=torch.float64, M=4, d=1)
        Kf = k_full(m)
        assert abs(float(conditional_mutual_information(m, [0], [3], [1, 2], Kf))) < 1e-8
        assert abs(float(conditional_mutual_information(m, [0], [2], [1], Kf))) < 1e-8


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_conditioning_reduces_covariance_psd_order(dtype):
    m = _model(random_dag, seed=2, dtype=dtype, M=5)
    Kf = k_full(m)
    A, B = [3], [0, 1]
    KAA = marginal(m, A, Kf)
    KA_B = conditional_covariance(m, A, B, Kf)
    diff = KAA - KA_B
    assert torch.linalg.eigvalsh(0.5 * (diff + diff.mH)).min().real > PSD_TOL


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_more_conditioning_reduces_further(dtype):
    m = _model(random_dag, seed=3, dtype=dtype, M=6)
    Kf = k_full(m)
    A, B, C = [5], [0], [1]
    cov_AB = conditional_covariance(m, A, B, Kf)
    cov_ABC = conditional_covariance(m, A, list(B) + list(C), Kf)
    diff = cov_AB - cov_ABC
    assert torch.linalg.eigvalsh(0.5 * (diff + diff.mH)).min().real > PSD_TOL


def test_data_processing_inequality_chain():
    # Markov chain 0->1->2: I(0;2) <= min(I(0;1), I(1;2))
    for seed in range(6):
        m = _model(chain_dag, seed=seed, dtype=torch.float64, M=3, d=2)
        Kf = k_full(m)
        i02 = float(mutual_information(m, [0], [2], Kf))
        i01 = float(mutual_information(m, [0], [1], Kf))
        i12 = float(mutual_information(m, [1], [2], Kf))
        assert i02 <= min(i01, i12) + 1e-9


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_mi_chain_rule(dtype):
    # I(A; B,C) = I(A;B) + I(A;C|B)
    m = _model(random_dag, seed=4, dtype=dtype, M=5)
    Kf = k_full(m)
    A, B, C = [0], [2], [4]
    lhs = mutual_information(m, A, list(B) + list(C), Kf)
    rhs = mutual_information(m, A, B, Kf) + conditional_mutual_information(m, A, C, B, Kf)
    assert abs(float(lhs - rhs)) < ATOL_INVARIANT


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_mi_nonnegative_random(dtype):
    for seed in range(5):
        m = _model(random_dag, seed=seed, dtype=dtype, M=5)
        Kf = k_full(m)
        assert float(mutual_information(m, [0], [4], Kf)) > -1e-10
