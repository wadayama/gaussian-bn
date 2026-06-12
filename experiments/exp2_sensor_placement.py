"""Experiment 2: sensor placement and edge identifiability.

Covers the inference-note Theme C / training-note Experiment 4. On a fixed
6-node DAG we ask: given a budget of observable nodes, which nodes should we
observe to best identify the edge parameters? We use the pullback Fisher metric
G^{(O)} over the edges and the D-optimal (logdet) and E-optimal (min-eigenvalue)
criteria. As the budget grows, the Fisher rank (number of identifiable edge
directions) and both criteria improve; greedy selection matches exhaustive on
this instance.

Results -> results/exp2.json.  Run: uv run python experiments/exp2_sensor_placement.py
"""

from __future__ import annotations

import json
import os

import torch

import gaussian_bn as gbn
from gaussian_bn.design import d_optimal_score, e_optimal_score

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp2.json")
SEED = 20260610


def model():
    # 6 scalar nodes: 0->1,0->2,1->3,2->4,3->5,4->5 (a diamond-ish lattice)
    dims = [1, 1, 1, 1, 1, 1]
    edges = {(0, 1): [[0.9]], (0, 2): [[-0.7]], (1, 3): [[1.1]],
             (2, 4): [[0.8]], (3, 5): [[0.6]], (4, 5): [[-0.9]]}
    noise = [[[1.0]], [[0.4]], [[0.5]], [[0.3]], [[0.6]], [[0.35]]]
    return gbn.GaussianDAG(dims, edges, noise)


def metrics_for_set(m, edge_params, observed, eps=1e-9):
    G, w, U = gbn.edge_fisher(m, edge_params, observed)
    wmax = float(w.max())
    rank = int((w > 1e-8 * wmax).sum())
    return {
        "logdet": float(d_optimal_score(G, eps=eps)),
        "lambda_min": float(e_optimal_score(G, eps=eps)),
        "rank": rank,
        "eigenvalues": [float(x) for x in w],
    }


def main():
    m = model()
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    q = len(edges)
    cands = [0, 1, 2, 3, 4, 5]

    # D-optimal greedy sweep over budgets, recording the chosen set's metrics
    d_curve = []
    for budget in range(1, 7):
        res = gbn.sensor_placement(m, edges, cands, budget, criterion="d", method="greedy")
        mt = metrics_for_set(m, edges, list(res.chosen))
        d_curve.append({"budget": budget, "chosen": list(res.chosen), **mt})

    # E-optimal greedy sweep
    e_curve = []
    for budget in range(1, 7):
        res = gbn.sensor_placement(m, edges, cands, budget, criterion="e", method="greedy")
        mt = metrics_for_set(m, edges, list(res.chosen))
        e_curve.append({"budget": budget, "chosen": list(res.chosen), **mt})

    # greedy vs exhaustive at budget 3 (both criteria)
    cmp = {}
    for crit in ("d", "e"):
        g = gbn.sensor_placement(m, edges, cands, 3, criterion=crit, method="greedy")
        ex = gbn.sensor_placement(m, edges, cands, 3, criterion=crit, method="exhaustive")
        cmp[crit] = {
            "greedy_chosen": list(g.chosen), "greedy_score": g.score,
            "exhaustive_chosen": list(ex.chosen), "exhaustive_score": ex.score,
            "match": abs(g.score - ex.score) < 1e-6,
        }

    results = {
        "seed": SEED, "q_edges": q, "edges": [list(e) for e in edges],
        "candidate_nodes": cands,
        "d_curve": d_curve, "e_curve": e_curve, "greedy_vs_exhaustive": cmp,
        "full_rank_budget": next((r["budget"] for r in d_curve if r["rank"] == q), None),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("budget  D-chosen        logdetG   lambda_min   rank")
    for r in d_curve:
        print(f"  {r['budget']}     {str(r['chosen']):16s} {r['logdet']:8.3f} {r['lambda_min']:10.4f}   {r['rank']}/{q}")
    print("greedy vs exhaustive (budget 3):")
    for crit, c in cmp.items():
        print(f"  {crit}: greedy {c['greedy_chosen']} ({c['greedy_score']:.3f}) "
              f"exhaustive {c['exhaustive_chosen']} ({c['exhaustive_score']:.3f}) match={c['match']}")
    print(f"first budget reaching full edge rank {q}: {results['full_rank_budget']}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
