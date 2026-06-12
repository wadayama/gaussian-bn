"""Tests for Fisher metric, edge_fisher, and identifiability_report (gauge rank)."""

from __future__ import annotations

import torch

from gaussian_bn.identifiability import (
    edge_fisher,
    fisher_metric,
    identifiability_report,
)
from gaussian_bn.inference import marginal
from gaussian_bn.model import GaussianDAG

from _helpers import diamond_dag, is_hermitian


def _cell(x):
    """A 1x1 float64 tensor from a python float or a 0-d/1-elem tensor (grad-safe)."""
    if isinstance(x, torch.Tensor):
        return x.reshape(1, 1)
    return torch.tensor([[float(x)]], dtype=torch.float64)


def latent_scale_model(a, b, s, s1=0.3, s2=0.4):
    edges = {(0, 1): _cell(a), (0, 2): _cell(b)}
    noise = [_cell(s), _cell(s1), _cell(s2)]
    return GaussianDAG([1, 1, 1], edges, noise, dtype=torch.float64, validate=False)


def test_fisher_metric_symmetric_psd():
    G, w, U = edge_fisher(diamond_model(), [(0, 1), (1, 3)], [0, 3])
    assert is_hermitian(G, atol=1e-10)
    assert w.min() > -1e-10


def diamond_model():
    dims, edges, noise = diamond_dag(seed=5, dtype=torch.float64)
    return GaussianDAG(dims, edges, noise)


def test_latent_scale_gauge_rank():
    # eta=(a,b,s): scale gauge a->a/t, b->b/t, s->t^2 s leaves observed cov fixed
    a0, b0, s0 = 1.2, 0.8, 1.0
    observed = [1, 2]

    def K_of_eta(eta):
        m = latent_scale_model(eta[0], eta[1], eta[2])
        return marginal(m, observed)

    eta0 = torch.tensor([a0, b0, s0], dtype=torch.float64)
    G, w, U = fisher_metric(K_of_eta, eta0)
    wmax = float(w.max())
    rank = int((w > 1e-8 * wmax).sum())
    assert rank == 2                                  # one gauge direction
    gauge = torch.tensor([-a0, -b0, 2 * s0], dtype=torch.float64)
    gauge = gauge / torch.linalg.norm(gauge)
    null_vec = U[:, 0]
    assert abs(float(torch.dot(null_vec, gauge))) > 0.999


def test_fixed_s_is_identifiable():
    a0, b0, s0 = 1.2, 0.8, 1.0
    observed = [1, 2]

    def K_of_eta(eta):  # only (a, b) free, s fixed
        m = latent_scale_model(eta[0], eta[1], s0)
        return marginal(m, observed)

    G, w, U = fisher_metric(K_of_eta, torch.tensor([a0, b0], dtype=torch.float64))
    assert int((w > 1e-8 * float(w.max())).sum()) == 2


def test_edge_fisher_autograd_vs_fd():
    m = diamond_model()
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    G_ad, _, _ = edge_fisher(m, edges, [0, 3], method="autograd")
    G_fd, _, _ = edge_fisher(m, edges, [0, 3], method="fd", fd_eps=1e-6)
    assert torch.linalg.norm(G_ad - G_fd) / torch.linalg.norm(G_ad) < 1e-4


def test_fully_observed_edges_identifiable():
    m = diamond_model()
    rep = identifiability_report(m, [(0, 1), (0, 2), (1, 3), (2, 3)], [0, 1, 2, 3])
    assert rep.identifiable
    assert rep.rank == rep.q
    assert rep.condition_number < float("inf")


def test_report_flags_gauge():
    a0, b0, s0 = 1.2, 0.8, 1.0
    # hidden node 0, observe {1,2}: with both edges free and noise fixed, identifiable;
    # the gauge appears only when the latent scale is also free -> use report on a model
    # where the two outgoing edges share the latent: here just check structure of report.
    m = latent_scale_model(a0, b0, s0)
    rep = identifiability_report(m, [(0, 1), (0, 2)], [1, 2])
    assert rep.q == 2
    assert rep.eigenvalues.numel() == 2
    assert rep.per_param_sensitivity.numel() == 2
