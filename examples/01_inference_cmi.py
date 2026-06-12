"""Example 1: build a Gaussian BN and query (conditional) mutual information.

Demonstrates marginalization, conditioning, MI, CMI, and how conditional
independence shows up as a zero CMI (d-separation).

Run:  uv run python examples/01_inference_cmi.py
"""

from __future__ import annotations

import gaussian_bn as gbn


def main() -> None:
    # Diamond DAG: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (scalar nodes).
    m = gbn.GaussianDAG(
        dims=[1, 1, 1, 1],
        edges={(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]},
        noise=[[[1.0]], [[0.4]], [[0.5]], [[0.3]]],
    )
    Kf = gbn.k_full(m)  # global covariance K_all (cache it and pass to queries)

    print("Marginal covariance of {0, 3}:")
    print(gbn.marginal(m, [0, 3], Kf))

    print("\nMutual information:")
    print(f"  I(V0; V3)      = {float(gbn.mutual_information(m, [0], [3], Kf)):.4f} nats")

    print("\nConditional mutual information:")
    # In this diamond, V1 and V2 are conditionally independent given their common
    # parent V0, so I(V1; V2 | V0) = 0 (a d-separation relation).
    cmi_indep = float(gbn.conditional_mutual_information(m, [1], [2], [0], Kf))
    # But V1 and V2 become dependent given the collider V3 (explaining away).
    cmi_dep = float(gbn.conditional_mutual_information(m, [1], [2], [3], Kf))
    print(f"  I(V1; V2 | V0) = {cmi_indep:.4f} nats   (conditionally independent)")
    print(f"  I(V1; V2 | V3) = {cmi_dep:.4f} nats   (collider opens the path)")

    print("\nPosterior of V3 given V0 = 1.0:")
    mean = gbn.conditional_mean(m, [3], [0], b=[1.0], Kf=Kf)
    cov = gbn.conditional_covariance(m, [3], [0], Kf)
    print(f"  E[V3 | V0=1] = {float(mean):.4f},  Var(V3 | V0) = {float(cov):.4f}")


if __name__ == "__main__":
    main()
