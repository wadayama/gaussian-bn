"""Observation / sensor design from the edge-Fisher information.

Given a set of edge parameters of interest and a pool of candidate observable
nodes, choose an observation set ``O`` (with ``|O| <= budget``) that maximizes an
optimality criterion of the pullback Fisher information ``G^{(O)}``:

* D-optimal: ``max_O  logdet(G^{(O)} + eps I)`` (volume of the information ellipsoid);
* E-optimal: ``max_O  lambda_min(G^{(O)} + eps I)`` (worst-case parameter direction).

Both a ``greedy`` forward selection and an ``exhaustive`` search are provided; on
small instances they agree, and greedy is monotone because adding an observation
never decreases the information.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

import torch

from gaussian_bn.identifiability import edge_fisher
from gaussian_bn.linalg import hermitianize, logdet_hpd
from gaussian_bn.model import GaussianDAG


def d_optimal_score(G: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """``logdet(G + eps I)`` (D-optimality)."""
    return logdet_hpd(G, jitter=eps)


def e_optimal_score(G: torch.Tensor, *, eps: float = 1e-9) -> torch.Tensor:
    """``lambda_min(G) + eps`` (E-optimality)."""
    return torch.linalg.eigvalsh(hermitianize(G)).min().real + eps


@dataclass
class SensorPlacement:
    """Result of :func:`sensor_placement`.

    Attributes:
        chosen: The selected observation node set.
        score: The optimality score at ``chosen``.
        trace: Per-step scores (greedy) or ``[score]`` (exhaustive).
    """

    chosen: tuple[int, ...]
    score: float
    trace: list[float]


def _score(model: GaussianDAG, edge_params, observed, criterion, eps, jitter) -> float:
    G, _, _ = edge_fisher(model, edge_params, observed, jitter=jitter)
    fn = d_optimal_score if criterion == "d" else e_optimal_score
    return float(fn(G, eps=eps))


def sensor_placement(
    model: GaussianDAG,
    edge_params: Sequence[tuple[int, int]],
    candidate_nodes: Sequence[int],
    budget: int,
    *,
    criterion: Literal["d", "e"] = "d",
    method: Literal["greedy", "exhaustive"] = "greedy",
    eps: float = 1e-9,
    jitter: float = 0.0,
) -> SensorPlacement:
    """Choose up to ``budget`` observation nodes maximizing a Fisher optimality criterion.

    Args:
        model: The Gaussian BN.
        edge_params: Edge parameters whose identifiability we care about.
        candidate_nodes: Pool of observable node indices to choose from.
        budget: Maximum number of nodes to select (``1 <= budget <= len(candidate_nodes)``).
        criterion: ``"d"`` (log-det) or ``"e"`` (min-eigenvalue).
        method: ``"greedy"`` forward selection or ``"exhaustive"`` search.
        eps: Diagonal floor on ``G`` inside the score.
        jitter: Diagonal shift used in the Fisher computation.

    Returns:
        A :class:`SensorPlacement`.
    """
    cands = list(candidate_nodes)
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}.")
    if budget > len(cands):
        raise ValueError(f"budget {budget} exceeds the {len(cands)} candidate nodes.")

    if method == "exhaustive":
        best_set: tuple[int, ...] | None = None
        best_score = -float("inf")
        for combo in combinations(cands, budget):
            s = _score(model, edge_params, list(combo), criterion, eps, jitter)
            if s > best_score:
                best_score, best_set = s, combo
        assert best_set is not None
        return SensorPlacement(best_set, best_score, [best_score])

    # greedy forward selection
    chosen: list[int] = []
    remaining = list(cands)
    trace: list[float] = []
    last_score = -float("inf")
    for _ in range(budget):
        best_node = None
        best_score = -float("inf")
        for n in remaining:
            s = _score(model, edge_params, chosen + [n], criterion, eps, jitter)
            if s > best_score:
                best_score, best_node = s, n
        chosen.append(best_node)
        remaining.remove(best_node)
        trace.append(best_score)
        last_score = best_score
    return SensorPlacement(tuple(chosen), last_score, trace)
