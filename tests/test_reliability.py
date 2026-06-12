"""Tests for gaussian_bn.reliability: Slepian–Bangs Fisher and Cramér–Rao bound."""

from __future__ import annotations

import math

import pytest
import torch

from gaussian_bn.identifiability import edge_fisher
from gaussian_bn.inference import sample
from gaussian_bn.model import GaussianDAG
from gaussian_bn.reliability import crb, crb_report, parameter_fisher
from gaussian_bn.training import fit_local_regression

from _helpers import gen


def chain():
    return GaussianDAG([1, 1, 1], {(0, 1): [[0.9]], (1, 2): [[-0.7]]},
                       [[[1.0]], [[0.5]], [[0.4]]])


def latent():
    # hidden node 0, observed children 1, 2
    return GaussianDAG([1, 1, 1], {(0, 1): [[1.2]], (0, 2): [[0.8]]},
                       [[[1.0]], [[0.3]], [[0.4]]])


def test_parameter_fisher_matches_edge_fisher():
    m = chain()
    F1, w1, U1, labels = parameter_fisher(m, [0, 1, 2], free_edges=[(0, 1), (1, 2)])
    F2, w2, U2 = edge_fisher(m, [(0, 1), (1, 2)], [0, 1, 2])
    assert torch.allclose(F1, F2, atol=1e-10)
    assert labels == ["A[0->1][0,0]", "A[1->2][0,0]"]


def test_crb_equals_fisher_inverse_full_rank():
    m = chain()
    N = 400
    F, w, U, _ = parameter_fisher(m, [0, 1, 2], free_edges=[(0, 1), (1, 2)])
    crb_mat = crb(m, [0, 1, 2], N, free_edges=[(0, 1), (1, 2)])
    assert torch.allclose(crb_mat, torch.linalg.inv(N * F), atol=1e-10)


def test_crb_scales_with_N():
    m = chain()
    c1 = crb(m, [0, 1, 2], 100, free_edges=[(0, 1), (1, 2)])
    c2 = crb(m, [0, 1, 2], 400, free_edges=[(0, 1), (1, 2)])
    assert torch.allclose(4.0 * c2, c1, atol=1e-10)   # CRB ∝ 1/N


def test_crb_report_fields_and_ci():
    m = chain()
    rep = crb_report(m, [0, 1, 2], 400, free_edges=[(0, 1), (1, 2)], confidence=0.95)
    assert rep.identifiable and rep.rank == rep.q == 2
    assert len(rep.labels) == 2
    # theta_hat defaults to the model's edge values
    assert torch.allclose(rep.theta_hat, torch.tensor([0.9, -0.7], dtype=torch.float64), atol=1e-10)
    # CI half-width = z * SE with z ≈ 1.96 for 95%
    z = math.sqrt(2.0) * float(torch.special.erfinv(torch.tensor(0.95)))
    assert torch.allclose(rep.ci_half_widths, z * rep.standard_errors, atol=1e-12)
    assert rep.confidence_intervals.shape == (2, 2)
    assert "CRB reliability report" in rep.summary()


def test_crb_gauge_flags_non_identifiable():
    # estimating edges + all variances of a partially observed latent model is
    # non-identifiable (latent scale gauge): SE must be inf and identifiable False.
    m = latent()
    rep = crb_report(m, [1, 2], 1000, free_edges=[(0, 1), (0, 2)], noise_param="logdiag")
    assert not rep.identifiable
    assert rep.rank < rep.q
    assert bool(torch.isinf(rep.standard_errors).any())


def test_crb_edges_identifiable_when_noise_known():
    # with the noise (incl. latent variance) fixed, the two edges ARE identifiable
    m = latent()
    rep = crb_report(m, [1, 2], 1000, free_edges=[(0, 1), (0, 2)], noise_param="fixed")
    assert rep.identifiable and rep.rank == 2
    assert torch.isfinite(rep.standard_errors).all()


@pytest.mark.slow
def test_crb_matches_empirical_mle():
    # the MLE attains the CRB: empirical Cov(theta_hat) ≈ (N F)^{-1}
    m = chain()
    N, T = 400, 2000
    crb_mat = crb(m, [0, 1, 2], N, free_edges=[(0, 1), (1, 2)])
    ahat = torch.zeros(T, 2, dtype=torch.float64)
    for t in range(T):
        X = sample(m, N, gen(5000 + t))
        eh, _ = fit_local_regression(m, X)
        ahat[t, 0] = eh[(0, 1)].item()
        ahat[t, 1] = eh[(1, 2)].item()
    emp_cov = torch.cov(ahat.T)
    rel = float(torch.linalg.norm(emp_cov - crb_mat) / torch.linalg.norm(crb_mat))
    assert rel < 0.1                                   # within Monte-Carlo error
