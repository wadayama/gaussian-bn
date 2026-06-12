"""Tests for non-zero-mean (affine offset) models — Phase 1.

The mean decouples from the covariance: the K-recursion / MI / CMI are unchanged,
and a parallel mean recursion m_j = c_j + sum_i A_ji m_i carries the offsets.
Zero-mean models are fully backward-compatible (has_mean=False, mean_all=0).
"""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import (
    conditional_mean,
    mutual_information,
    sample,
)
from gaussian_bn.intervention import do_hard
from gaussian_bn.model import GaussianDAG, k_full, mean_all
from gaussian_bn.reliability import crb, parameter_fisher
from gaussian_bn.training import (
    ObsPattern,
    fit_gradient_custom,
    fit_local_regression,
    marginal_likelihood,
)

from _helpers import gen, is_psd


def chain_offsets(c=(2.0, -1.0, 0.5)):
    return GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                       [[[1.0]], [[0.5]], [[0.4]]],
                       mean=[torch.tensor([x]) for x in c])


# --------------------------------------------------------------------------
# backward compatibility
# --------------------------------------------------------------------------
def test_zero_mean_default_backward_compat():
    m = GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                    [[[1.0]], [[0.5]], [[0.4]]])
    assert m.has_mean is False
    assert torch.all(mean_all(m) == 0)
    # conditional_mean reduces to the zero-mean formula
    Kf = k_full(m)
    got = conditional_mean(m, [2], [0], torch.tensor([1.0]), Kf)
    assert torch.allclose(got, Kf[2:3, 0:1] / Kf[0, 0] * 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# mean recursion
# --------------------------------------------------------------------------
def test_mean_all_matches_closed_form():
    m = chain_offsets()
    A = torch.zeros(3, 3, dtype=torch.float64)
    A[1, 0] = 0.8
    A[2, 1] = 0.9
    ref = torch.linalg.inv(torch.eye(3, dtype=torch.float64) - A) @ torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    assert torch.allclose(mean_all(m), ref, atol=1e-12)
    assert m.has_mean


def test_sample_mean_matches():
    m = chain_offsets()
    X = sample(m, 200_000, gen(0))
    assert torch.linalg.norm(X.mean(0) - mean_all(m)) < 1e-2


# --------------------------------------------------------------------------
# the covariance side is invariant to the mean
# --------------------------------------------------------------------------
def test_mi_invariant_to_mean():
    m = chain_offsets()
    m0 = GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                     [[[1.0]], [[0.5]], [[0.4]]])
    assert abs(float(mutual_information(m, [0], [2]) - mutual_information(m0, [0], [2]))) < 1e-12
    assert torch.allclose(k_full(m), k_full(m0), atol=1e-12)


# --------------------------------------------------------------------------
# conditioning mean-shift
# --------------------------------------------------------------------------
def test_conditional_mean_shift():
    m = chain_offsets()
    Kf = k_full(m)
    ma = mean_all(m)
    v = torch.tensor([5.0], dtype=torch.float64)
    got = conditional_mean(m, [2], [0], v, Kf)
    exp = ma[2] + Kf[2, 0] / Kf[0, 0] * (v - ma[0])
    assert torch.allclose(got, exp, atol=1e-12)


# --------------------------------------------------------------------------
# point intervention do(V_j = u)
# --------------------------------------------------------------------------
def test_point_intervention_propagates_mean():
    m = chain_offsets()
    mdo = do_hard(m, 1, value=torch.tensor([10.0]))
    md = mean_all(mdo)
    assert abs(float(md[1]) - 10.0) < 1e-12              # node fixed to the value
    assert abs(float(md[2]) - (0.5 + 0.9 * 10.0)) < 1e-12  # downstream propagation
    # the intervened node is deterministic; V2 keeps only its own innovation noise
    Kdo = k_full(mdo)
    assert abs(float(Kdo[2, 2]) - 0.4) < 1e-12
    assert abs(float(Kdo[1, 1])) < 1e-12                 # deterministic node: zero variance
    # original model untouched
    assert float(mean_all(m)[1]) != 10.0


def test_point_intervention_breaks_upstream_dependence():
    # do(V1) makes V0 and V2 independent (mutilated chain 0  1->2 with 1 fixed)
    m = chain_offsets()
    mdo = do_hard(m, 1, value=torch.tensor([3.0]))
    assert abs(float(mutual_information(mdo, [0], [2]))) < 1e-9


# --------------------------------------------------------------------------
# centered likelihood
# --------------------------------------------------------------------------
def test_marginal_likelihood_centered():
    m = chain_offsets()
    X = sample(m, 5000, gen(1))
    observed = [0, 2]
    oi = m.node_index(observed)
    # mean-model NLL on data == zero-mean-model NLL on centered data
    nll_mean = float(marginal_likelihood(m, [ObsPattern((0, 2), X[:, oi])]))
    m0 = GaussianDAG(m.dims, dict(m.edges), list(m.noise))    # same covariance, zero mean
    ma = mean_all(m)
    Xc = X[:, oi] - ma[oi]
    nll_zero = float(marginal_likelihood(m0, [ObsPattern((0, 2), Xc)]))
    assert abs(nll_mean - nll_zero) < 1e-9


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_mean_wrong_shape_raises():
    with pytest.raises(ValueError):
        GaussianDAG([2, 1], {(0, 1): [[1.0, 0.0]]}, [torch.eye(2), torch.eye(1)],
                    mean=[torch.zeros(3), torch.zeros(1)])   # node 0 mean wrong length


# --------------------------------------------------------------------------
# Phase 2: learning the mean and the Slepian–Bangs mean term
# --------------------------------------------------------------------------
def test_affine_mle_recovery():
    m = chain_offsets()
    X = sample(m, 200_000, gen(2))
    edges, noise, mean = fit_local_regression(m, X, fit_mean=True)
    for k in m.edges:
        assert float(torch.linalg.norm(edges[k] - m.edges[k])) < 0.02
    rec = GaussianDAG(m.dims, edges, noise, mean=mean)
    assert float(torch.linalg.norm(mean_all(rec) - mean_all(m))) < 0.02


def test_gradient_learns_offsets_from_zero():
    # learn the node offsets c (edges/noise fixed) starting from zero
    m = chain_offsets()
    X = sample(m, 50_000, gen(3))
    c = [torch.zeros(1, dtype=torch.float64, requires_grad=True) for _ in range(3)]

    def build():
        return GaussianDAG(m.dims, dict(m.edges), list(m.noise), mean=c, validate=False)

    hist = fit_gradient_custom(c, build, [ObsPattern((0, 1, 2), X)],
                               optimizer="lbfgs", lr=1.0, num_iters=200)
    assert hist[-1] < hist[0]
    with torch.no_grad():
        assert float(torch.linalg.norm(mean_all(build()) - mean_all(m))) < 0.05


def test_crb_mean_term_tightens_and_off_by_default():
    # affine model; by default (offsets estimated) the edge CRB is covariance-only
    m = chain_offsets()
    m0 = GaussianDAG(m.dims, dict(m.edges), list(m.noise))   # zero-mean twin
    F_def, *_ = parameter_fisher(m, [0, 1, 2], free_edges=[(0, 1), (1, 2)])
    F_cov, *_ = parameter_fisher(m0, [0, 1, 2], free_edges=[(0, 1), (1, 2)])
    assert torch.allclose(F_def, F_cov, atol=1e-10)         # mean term off by default
    # with known offsets the mean term ADDS information (F grows in PSD order)
    F_full, *_ = parameter_fisher(m, [0, 1, 2], free_edges=[(0, 1), (1, 2)],
                                  include_mean=True)
    assert is_psd(F_full - F_def, tol=-1e-10)
    assert float(torch.linalg.norm(F_full - F_def)) > 1e-6   # genuinely larger
