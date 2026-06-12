"""Example 2: estimate BN parameters from data (full and partial observation).

Full observation -> closed-form node-wise MLE recovers the true parameters.
Hidden middle nodes -> EM on the K-recursion recovers the observed covariance,
and the Fisher metric reveals that the edge parameters are only identifiable up
to a latent gauge.

Run:  uv run python examples/02_estimation_em.py
"""

from __future__ import annotations

import torch

import gaussian_bn as gbn


def main() -> None:
    torch.manual_seed(0)
    m = gbn.GaussianDAG(
        dims=[1, 1, 1, 1],
        edges={(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]},
        noise=[[[1.0]], [[0.4]], [[0.5]], [[0.3]]],
    )
    X = gbn.sample(m, 200_000, torch.Generator().manual_seed(1))

    # --- full observation: closed-form MLE ---
    edges_hat, noise_hat = gbn.fit_local_regression(m, X)
    edge_err = max(float(torch.linalg.norm(edges_hat[k] - m.edges[k])) for k in m.edges)
    print(f"Full-observation MLE: max edge error = {edge_err:.2e}")

    # --- hidden middle nodes {1, 2}, observe only {0, 3}: EM ---
    observed = [0, 3]
    fitted, nll_history = gbn.em_fit(m, X, observed, num_iters=60)
    oi = m.node_index(observed)
    S = (X[:, oi].mH @ X[:, oi]) / X.shape[0]
    K_OO_hat = gbn.k_full(fitted)[oi][:, oi]
    rel = float(torch.linalg.norm(K_OO_hat - S) / torch.linalg.norm(S))
    print(f"EM (hidden 1,2): NLL {nll_history[0]:.4f} -> {nll_history[-1]:.4f}, "
          f"observed-cov rel. error = {rel:.2e}")

    # --- identifiability of the edge parameters from {0, 3} ---
    report = gbn.identifiability_report(
        m, edge_params=[(0, 1), (0, 2), (1, 3), (2, 3)], observed=observed,
    )
    print(f"Edge identifiability from {observed}: identifiable={report.identifiable}, "
          f"rank={report.rank}/{report.q}  "
          f"(rank deficit = {report.q - report.rank} latent gauge directions)")


if __name__ == "__main__":
    main()
