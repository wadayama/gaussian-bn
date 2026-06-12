"""Tests for gaussian_bn.intervention: hard/soft do-operations."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.inference import mutual_information
from gaussian_bn.intervention import (
    counterfactual,
    do_cmi,
    do_covariance,
    do_hard,
    do_soft,
)
from gaussian_bn.model import GaussianDAG, k_full

from _helpers import ATOL_EXACT, diamond_dag, relerr_fro


def _model():
    dims, edges, noise = diamond_dag(seed=5, dtype=torch.float64)
    return GaussianDAG(dims, edges, noise)


def test_do_hard_returns_new_model_unchanged_original():
    m = _model()
    Kf_before = k_full(m).clone()
    m2 = do_hard(m, 3)
    assert m2 is not m
    # original untouched
    assert torch.allclose(k_full(m), Kf_before, atol=ATOL_EXACT)
    assert (1, 3) in m.edges and (1, 3) not in m2.edges


def test_do_hard_equals_manual_edge_deletion():
    m = _model()
    node = 3
    cov = torch.tensor([[0.7, 0.1], [0.1, 0.9]], dtype=torch.float64)
    auto = do_covariance(m, node, cov=cov)
    # manual: drop incoming edges to node 3, set its noise to cov
    edges = {k: v for k, v in m.edges.items() if k[1] != node}
    noise = list(m.noise); noise[node] = cov
    manual = k_full(GaussianDAG(m.dims, edges, noise))
    assert relerr_fro(auto, manual) < 1e-12


def test_do_soft_replaces_mechanism():
    m = _model()
    newA = torch.zeros(2, 2, dtype=torch.float64)
    m2 = do_soft(m, 3, edges={1: newA})
    # manual model with the replaced edge
    edges = dict(m.edges); edges[(1, 3)] = newA
    manual = k_full(GaussianDAG(m.dims, edges, list(m.noise)))
    assert relerr_fro(k_full(m2), manual) < 1e-12


def test_do_hard_breaks_dependence_chain():
    # chain 0 -> 1 -> 2: intervening on 1 makes 0 and 2 independent
    m = GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                    [[[1.0]], [[0.5]], [[0.5]]])
    assert float(mutual_information(m, [0], [2])) > 1e-3
    m_do = do_hard(m, 1)
    assert abs(float(mutual_information(m_do, [0], [2]))) < 1e-10


def test_do_cmi_runs_on_intervened_model():
    m = _model()
    val = do_cmi(m, 3, [0], [1], [2])
    assert torch.isfinite(val)


def test_do_validation_errors():
    m = _model()
    with pytest.raises(ValueError):
        do_hard(m, 99)
    with pytest.raises(ValueError):
        # wrong-dimension cov triggers validation failure
        do_covariance(m, 3, cov=torch.eye(3, dtype=torch.float64))


# --------------------------------------------------------------------------
# counterfactuals (abduction-action-prediction)
# --------------------------------------------------------------------------
def _chain(mean=None):
    return GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                       [[[1.0]], [[0.5]], [[0.4]]], mean=mean)


def test_counterfactual_full_evidence_deterministic():
    m = _chain(mean=[torch.tensor([2.0]), torch.tensor([-1.0]), torch.tensor([0.5])])
    e = torch.tensor([3.0, 1.5, 0.7], dtype=torch.float64)
    cf = counterfactual(m, [0, 1, 2], e, {0: torch.tensor([5.0])}, [2])
    # full evidence -> deterministic; ΔV2 = a21 a10 (5 - 3)
    assert abs(float(cf) - (0.7 + 0.9 * 0.8 * 2.0)) < 1e-10


def test_counterfactual_null_recovers_factual():
    m = _chain(mean=[torch.tensor([2.0]), torch.tensor([-1.0]), torch.tensor([0.5])])
    e = torch.tensor([3.0, 1.5, 0.7], dtype=torch.float64)
    cf = counterfactual(m, [0, 1, 2], e, {0: torch.tensor([3.0])}, [2])  # do = factual
    assert abs(float(cf) - 0.7) < 1e-10


def test_counterfactual_zero_mean_model():
    m = _chain()  # zero-mean
    e = torch.tensor([1.0, 0.5, 0.3], dtype=torch.float64)
    cf = counterfactual(m, [0, 1, 2], e, {0: torch.tensor([2.0])}, [2])
    assert abs(float(cf) - (0.3 + 0.9 * 0.8 * 1.0)) < 1e-10
