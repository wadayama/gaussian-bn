"""Experiment 1: hidden-node estimation -- EM vs gradient training.

Covers the training-note Experiments 2 & 3: a diamond DAG with the two middle
nodes hidden, observed only at the root and sink. We fit the model three ways
(EM, Adam, LBFGS), compare the observed-data NLL trajectories and the recovered
observed covariance, and diagnose the latent edge gauge with the Fisher rank.

All numbers are written to results/exp1.json (the source of truth for the
summary). Run:  uv run python experiments/exp1_hidden_em.py
"""

from __future__ import annotations

import json
import os
import time

import torch

import gaussian_bn as gbn

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp1.json")
SEED = 20260610
torch.manual_seed(SEED)


def gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def true_model():
    dims = [1, 1, 1, 1]
    edges = {(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]}
    noise = [[[1.0]], [[0.4]], [[0.5]], [[0.3]]]
    return gbn.GaussianDAG(dims, edges, noise)


def main():
    m = true_model()
    observed = [0, 3]
    oi = m.node_index(observed)
    K_OO_true = gbn.k_full(m)[oi][:, oi].detach()

    N = 200_000
    X = gbn.sample(m, N, gen(SEED + 1))
    Y = X[:, oi]
    S = (Y.mH @ Y) / N                     # observed sample covariance (the data)
    sample_vs_true = float(torch.linalg.norm(S - K_OO_true) / torch.linalg.norm(K_OO_true))

    keys = [(0, 1), (0, 2), (1, 3), (2, 3)]

    def random_init(seed):
        g = gen(seed)
        edges = {k: torch.randn(1, 1, generator=g, dtype=torch.float64) for k in keys}
        noise = [torch.tensor([[0.7]], dtype=torch.float64) for _ in range(4)]
        return gbn.GaussianDAG(m.dims, edges, noise)

    def kOO(model):
        return gbn.k_full(model)[oi][:, oi].detach()

    def relerr(model):
        return float(torch.linalg.norm(kOO(model) - S) / torch.linalg.norm(S))

    pattern = [gbn.ObsPattern(tuple(observed), Y)]

    # --- EM ---
    t0 = time.perf_counter()
    em_model, em_hist = gbn.em_fit(random_init(SEED + 2), X, observed, num_iters=60)
    em_time = time.perf_counter() - t0

    # --- Adam ---
    t0 = time.perf_counter()
    adam_model, adam_hist = gbn.fit_gradient(
        random_init(SEED + 3), pattern, optimizer="adam", lr=0.05, num_iters=400)
    adam_time = time.perf_counter() - t0

    # --- LBFGS ---
    t0 = time.perf_counter()
    lbfgs_model, lbfgs_hist = gbn.fit_gradient(
        random_init(SEED + 4), pattern, optimizer="lbfgs", lr=1.0, num_iters=500)
    lbfgs_time = time.perf_counter() - t0

    # --- Fisher gauge diagnosis at the EM solution (edges, noise fixed) ---
    G, w, U = gbn.edge_fisher(em_model, keys, observed)
    wmax = float(w.max())
    rank = int((w > 1e-8 * wmax).sum())

    # NLL optimum value is logdet(S)+dim
    nll_opt = float(torch.linalg.slogdet(S).logabsdet + S.shape[0])

    results = {
        "seed": SEED, "N": N, "observed": observed,
        "K_OO_true": K_OO_true.tolist(), "sample_cov": S.tolist(),
        "sample_vs_true_relerr": sample_vs_true,
        "nll_optimum": nll_opt,
        "em": {"nll_history": em_hist, "nll_final": em_hist[-1],
               "relerr_KOO_vs_sample": relerr(em_model), "time_s": em_time,
               "iters": len(em_hist)},
        "adam": {"nll_history": adam_hist, "nll_final": adam_hist[-1],
                 "relerr_KOO_vs_sample": relerr(adam_model), "time_s": adam_time,
                 "iters": len(adam_hist)},
        "lbfgs": {"nll_final": lbfgs_hist[-1],
                  "relerr_KOO_vs_sample": relerr(lbfgs_model), "time_s": lbfgs_time,
                  "iters": len(lbfgs_hist)},
        "fisher_edge_eigenvalues": [float(x) for x in w],
        "fisher_edge_rank": rank, "fisher_edge_q": 4,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"EM    : NLL {em_hist[0]:.4f} -> {em_hist[-1]:.4f}  relerr_KOO={results['em']['relerr_KOO_vs_sample']:.2e}  {em_time:.2f}s")
    print(f"Adam  : NLL {adam_hist[0]:.4f} -> {adam_hist[-1]:.4f}  relerr_KOO={results['adam']['relerr_KOO_vs_sample']:.2e}  {adam_time:.2f}s")
    print(f"LBFGS : NLL ...    -> {lbfgs_hist[-1]:.4f}  relerr_KOO={results['lbfgs']['relerr_KOO_vs_sample']:.2e}  {lbfgs_time:.2f}s")
    print(f"NLL optimum (logdet S + dim) = {nll_opt:.4f}")
    print(f"Fisher edge rank = {rank}/4  eigvals={[round(float(x),4) for x in w]}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
