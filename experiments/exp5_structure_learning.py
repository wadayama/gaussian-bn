"""Experiment 5: structure learning by group-sparsity edge pruning.

Covers the training-note Experiment 5 / Theme E. Starting from a fully connected
"supergraph" candidate DAG (all edges i->j with i<j), we fit a VECTOR-valued
linear Gaussian BN by minimizing the K-recursion marginal likelihood plus a
GROUP-SPARSITY penalty on the edge matrices,
    min_eta  L(eta) + lambda * sum_{(i,j)} || A_{ji} ||_F ,
solved by proximal gradient: a gradient step on the smooth NLL followed by a
per-edge group soft-threshold that zeros an ENTIRE 2x2 edge block at once.

Nodes are 2-dimensional so ||A_{ji}||_F is a genuine block (group) norm, and the
sample size is small so the unregularized MLE (lambda=0) over-fits spurious
edges; group sparsity prunes them and BIC selects the structure.

The smooth loss is the library's marginal_likelihood (full observation here), so
the same machinery extends to hidden nodes by changing the observed set.

Results -> results/exp5.json.  Run: uv run python experiments/exp5_structure_learning.py
"""

from __future__ import annotations

import json
import math
import os

import torch

import gaussian_bn as gbn

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp5.json")
SEED = 20260610
torch.manual_seed(SEED)
D = 2
M = 5


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# True sparse DAG on 5 vector (2-d) nodes (topological order 0..4).
TRUE_EDGES = {
    (0, 1): [[0.8, 0.1], [-0.2, 0.7]],
    (1, 2): [[-0.6, 0.3], [0.2, -0.5]],
    (0, 3): [[0.5, -0.2], [0.1, 0.6]],
    (3, 4): [[0.7, 0.0], [0.3, 0.8]],
    (2, 4): [[0.4, 0.2], [-0.3, 0.5]],
}
NOISE_DIAG = [[1.0, 0.8], [0.4, 0.5], [0.5, 0.3], [0.6, 0.4], [0.3, 0.5]]
SUPER_EDGES = [(i, j) for j in range(M) for i in range(j)]   # all 10 candidate edges


def true_model():
    edges = {k: torch.tensor(v, dtype=torch.float64) for k, v in TRUE_EDGES.items()}
    noise = [torch.diag(torch.tensor(d, dtype=torch.float64)) for d in NOISE_DIAG]
    return gbn.GaussianDAG([D] * M, edges, noise)


def build(A, s):
    """Model from per-edge 2x2 leaves A and per-node log-noise-diag leaves s (length D)."""
    edges = {e: A[e] for e in SUPER_EDGES}
    noise = [torch.diag(torch.exp(s[j])) for j in range(M)]
    return gbn.GaussianDAG([D] * M, edges, noise, dtype=torch.float64, validate=False)


def fit_group_lasso(X, S, lam, *, A0=None, s0=None, lr=0.01, iters=2500):
    """Proximal gradient for group-lasso-penalized marginal likelihood."""
    A = {e: (A0[e].clone() if A0 else 0.01 * torch.randn(D, D, dtype=torch.float64,
                                                          generator=gen(SEED + hash(e) % 9999)))
            .requires_grad_(True) for e in SUPER_EDGES}
    if s0 is None:
        s = [torch.log(torch.diagonal(S)[j * D:(j + 1) * D].clamp(min=1e-3)).requires_grad_(True)
             for j in range(M)]
    else:
        s = [s0[j].clone().requires_grad_(True) for j in range(M)]
    pattern = [gbn.ObsPattern(tuple(range(M)), X)]

    for _ in range(iters):
        loss = gbn.marginal_likelihood(build(A, s), pattern)
        loss.backward()
        with torch.no_grad():
            for j in range(M):
                s[j] -= lr * s[j].grad
                s[j].grad.zero_()
            for e in SUPER_EDGES:
                Ae = A[e]
                Ae -= lr * Ae.grad                          # gradient step (smooth part)
                norm = float(torch.linalg.norm(Ae).detach()) # group soft-threshold (prox)
                shrink = max(0.0, 1.0 - lr * lam / (norm + 1e-12))
                Ae *= shrink
                Ae.grad.zero_()
    final_nll = float(gbn.marginal_likelihood(build(A, s), pattern).detach())
    return {e: A[e].detach() for e in SUPER_EDGES}, [si.detach() for si in s], final_nll


def refit_bic(support, X, N):
    """Unpenalized closed-form MLE on a fixed edge support; return (nll, BIC).

    Uses the library's full-observation local-regression MLE so the selected
    structure is scored without the group-lasso shrinkage bias.
    """
    support = list(support)
    if support:
        struct = gbn.GaussianDAG([D] * M, {e: torch.zeros(D, D, dtype=torch.float64) for e in support},
                                 [torch.eye(D, dtype=torch.float64) for _ in range(M)], validate=False)
        eh, nh = gbn.fit_local_regression(struct, X)
    else:
        eh, nh = {}, [(X[:, j * D:(j + 1) * D].mH @ X[:, j * D:(j + 1) * D]) / N for j in range(M)]
    refit = gbn.GaussianDAG([D] * M, eh, nh, validate=False)
    nll = float(gbn.marginal_likelihood(refit, [gbn.ObsPattern(tuple(range(M)), X)]).detach())
    # free params: D*D per edge + D*(D+1)/2 per (full) noise covariance
    k = D * D * len(support) + (D * (D + 1) // 2) * M
    return nll, N * nll + k * math.log(N)


def main():
    m = true_model()
    N = 400
    X = gbn.sample(m, N, gen(SEED + 1))
    S = (X.mH @ X) / N

    true_set = set(TRUE_EDGES)
    tol = 1e-6                       # prox gives exact zeros; present if ||A|| > tol

    lambdas = [0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 24.0, 36.0]
    path = []
    A0 = s0 = None
    for lam in lambdas:
        A, s, nll_pen = fit_group_lasso(X, S, lam, A0=A0, s0=s0)
        A0, s0 = A, s
        norms = {f"{i}_{j}": float(torch.linalg.norm(A[(i, j)])) for (i, j) in SUPER_EDGES}
        kept = {e for e in SUPER_EDGES if norms[f"{e[0]}_{e[1]}"] > tol}
        nll_refit, bic = refit_bic(kept, X, N)     # unpenalized refit score
        path.append({"lambda": lam, "nll_penalized": nll_pen, "nll_refit": nll_refit,
                     "bic": bic, "n_edges": len(kept),
                     "kept": sorted(list(kept)), "norms": norms})

    best = min(path, key=lambda r: r["bic"])
    kept = set(tuple(e) for e in best["kept"])
    tp = len(kept & true_set); fp = len(kept - true_set); fn = len(true_set - kept)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    results = {
        "seed": SEED, "N": N, "M": M, "node_dim": D,
        "true_edges": [list(e) for e in true_set],
        "super_edges": [list(e) for e in SUPER_EDGES],
        "spurious_edges": [list(e) for e in (set(SUPER_EDGES) - true_set)],
        "tol": tol, "lambdas": lambdas, "path": path,
        "selected": {
            "lambda": best["lambda"], "bic": best["bic"], "nll": best["nll_refit"],
            "kept": [list(e) for e in sorted(kept)],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "exact_recovery": kept == true_set,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"true edges     : {sorted(true_set)}")
    print(f"spurious (off) : {sorted(set(SUPER_EDGES) - true_set)}")
    print("  lambda  nll_pen  nll_refit  BIC(refit)  #edges  kept")
    for r in path:
        mark = " <-BIC" if r["lambda"] == best["lambda"] else ""
        print(f"  {r['lambda']:5.1f}  {r['nll_penalized']:.3f}   {r['nll_refit']:.3f}    "
              f"{r['bic']:.1f}   {r['n_edges']:2d}   {r['kept']}{mark}")
    s = results["selected"]
    print(f"\nBIC-selected lambda={s['lambda']}: TP={s['tp']} FP={s['fp']} FN={s['fn']} "
          f"precision={s['precision']:.2f} recall={s['recall']:.2f} exact={s['exact_recovery']}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
