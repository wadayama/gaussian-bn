"""Experiment 6: estimator reliability via the Slepian–Bangs Cramér–Rao bound.

The library computes the per-sample Fisher information analytically from the
K-recursion covariance (the Slepian–Bangs route), so the Cramér–Rao bound
CRB = (N F)^{-1} predicts the variance of any unbiased estimator. Here we verify
that the maximum-likelihood estimator attains it: over many independent trials,
the empirical covariance of the MLE matches the CRB, and the CRB-based standard
errors match the empirical standard errors.

We also show the partial-observation case, where a latent scale gauge makes a
parameter set non-identifiable and the CRB is (correctly) infinite.

Results -> results/exp6.json.  Run: uv run python experiments/exp6_crb_reliability.py
"""

from __future__ import annotations

import json
import os

import torch

import gaussian_bn as gbn

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp6.json")
SEED = 20260610
torch.set_default_dtype(torch.float64)


def gen(seed):
    g = torch.Generator(); g.manual_seed(seed); return g


def main():
    # chain 0 -> 1 -> 2 (scalar), two edges, all nodes observed
    m = gbn.GaussianDAG([1, 1, 1], {(0, 1): [[0.9]], (1, 2): [[-0.7]]},
                        [[[1.0]], [[0.5]], [[0.4]]])
    edge_keys = [(0, 1), (1, 2)]
    observed = [0, 1, 2]
    N, T = 400, 4000

    # --- CRB from the Slepian–Bangs Fisher (analytic, no sampling) ---
    rep = gbn.crb_report(m, observed, N, free_edges=edge_keys)
    crb_se = rep.standard_errors

    # --- empirical: MLE over T independent trials ---
    ahat = torch.zeros(T, 2)
    for t in range(T):
        X = gbn.sample(m, N, gen(SEED + 1000 + t))
        eh, _ = gbn.fit_local_regression(m, X)
        ahat[t, 0] = eh[(0, 1)].item()
        ahat[t, 1] = eh[(1, 2)].item()
    emp_cov = torch.cov(ahat.T)
    emp_se = torch.sqrt(torch.diagonal(emp_cov))
    bias = (ahat.mean(0) - torch.tensor([0.9, -0.7]))
    cov_rel_err = float(torch.linalg.norm(emp_cov - rep.crb) / torch.linalg.norm(rep.crb))

    # --- partial observation: latent scale gauge -> infinite CRB ---
    mlat = gbn.GaussianDAG([1, 1, 1], {(0, 1): [[1.2]], (0, 2): [[0.8]]},
                           [[[1.0]], [[0.3]], [[0.4]]])
    rep_gauge = gbn.crb_report(mlat, [1, 2], 1000,
                               free_edges=[(0, 1), (0, 2)], noise_param="logdiag")

    results = {
        "seed": SEED, "N": N, "T": T, "observed": observed,
        "fisher_per_sample": rep.fisher.tolist(),
        "crb_matrix": rep.crb.tolist(),
        "params": [
            {"label": rep.labels[a], "true": [0.9, -0.7][a],
             "bias": float(bias[a]),
             "crb_se": float(crb_se[a]), "emp_se": float(emp_se[a]),
             "emp_over_crb": float(emp_se[a] / crb_se[a]),
             "ci": [float(rep.confidence_intervals[a, 0]),
                    float(rep.confidence_intervals[a, 1])]}
            for a in range(2)
        ],
        "empirical_cov": emp_cov.tolist(),
        "cov_rel_err_vs_crb": cov_rel_err,
        "confidence": rep.confidence,
        "gauge_case": {
            "observed": [1, 2], "free_edges": [[0, 1], [0, 2]], "noise_param": "logdiag",
            "identifiable": rep_gauge.identifiable,
            "rank": rep_gauge.rank, "q": rep_gauge.q,
            "labels": rep_gauge.labels,
            "standard_errors": [("inf" if torch.isinf(s) else float(s))
                                for s in rep_gauge.standard_errors],
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(rep.summary())
    print(f"\n{'param':14s} {'bias':>10s} {'emp SE':>10s} {'CRB SE':>10s} {'emp/CRB':>8s}")
    for p in results["params"]:
        print(f"{p['label']:14s} {p['bias']:10.2e} {p['emp_se']:10.5f} "
              f"{p['crb_se']:10.5f} {p['emp_over_crb']:8.3f}")
    print(f"\nCov(theta_hat) vs CRB Frobenius rel.err = {cov_rel_err:.3f} "
          f"(Monte-Carlo error ~ {1.0 / T ** 0.5:.3f})")
    g = results["gauge_case"]
    print(f"\nPartial-observation gauge case: identifiable={g['identifiable']}, "
          f"rank {g['rank']}/{g['q']}, all SE = {g['standard_errors']}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
