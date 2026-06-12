"""Tests for gaussian_bn.design: D-/E-optimal sensor placement."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.design import (
    d_optimal_score,
    e_optimal_score,
    sensor_placement,
)
from gaussian_bn.identifiability import edge_fisher
from gaussian_bn.model import GaussianDAG

from _helpers import diamond_dag


def _model():
    dims, edges, noise = diamond_dag(seed=5, dtype=torch.float64)
    return GaussianDAG(dims, edges, noise)


EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_d_optimal_score_matches_logdet():
    m = _model()
    G, _, _ = edge_fisher(m, EDGES, [0, 3])
    ref = torch.linalg.slogdet(G + 1e-9 * torch.eye(G.shape[0], dtype=torch.float64)).logabsdet
    assert abs(float(d_optimal_score(G)) - float(ref)) < 1e-6


def test_e_optimal_score_matches_min_eig():
    m = _model()
    G, _, _ = edge_fisher(m, EDGES, [0, 3])
    ref = torch.linalg.eigvalsh(G).min() + 1e-9
    assert abs(float(e_optimal_score(G)) - float(ref)) < 1e-9


def test_greedy_d_optimal_monotone():
    m = _model()
    res = sensor_placement(m, EDGES, [0, 1, 2, 3], budget=3, criterion="d", method="greedy")
    diffs = [res.trace[t + 1] - res.trace[t] for t in range(len(res.trace) - 1)]
    assert min(diffs) > -1e-9            # adding a sensor never decreases information
    assert len(res.chosen) == 3


def test_greedy_matches_exhaustive_tiny():
    m = _model()
    for crit in ("d", "e"):
        greedy = sensor_placement(m, EDGES, [0, 1, 2, 3], budget=2,
                                  criterion=crit, method="greedy")
        exhaustive = sensor_placement(m, EDGES, [0, 1, 2, 3], budget=2,
                                      criterion=crit, method="exhaustive")
        # greedy reaches the exhaustive optimum on this tiny instance (same set,
        # so scores agree up to index-ordering round-off in the Fisher computation)
        assert abs(greedy.score - exhaustive.score) < 1e-5


def test_e_optimal_greedy_monotone():
    m = _model()
    res = sensor_placement(m, EDGES, [0, 1, 2, 3], budget=3, criterion="e", method="greedy")
    diffs = [res.trace[t + 1] - res.trace[t] for t in range(len(res.trace) - 1)]
    assert min(diffs) > -1e-9


def test_full_observation_identifies_all_edges():
    m = _model()
    res = sensor_placement(m, EDGES, [0, 1, 2, 3], budget=4, criterion="d")
    assert set(res.chosen) == {0, 1, 2, 3}


def test_design_validation():
    m = _model()
    with pytest.raises(ValueError):
        sensor_placement(m, EDGES, [0, 1], budget=3)   # budget > candidates
    with pytest.raises(ValueError):
        sensor_placement(m, EDGES, [0, 1], budget=0)
