"""Example 4: affine model, point intervention, and a counterfactual.

A causal chain X -> M -> Y with non-zero node offsets. We observe a unit's
(X, M, Y), then ask the counterfactual question: "had X been different, what
would Y have been?" — holding the unit's latent noise fixed (abduction).

Run:  uv run python examples/04_counterfactual.py
"""

from __future__ import annotations

import torch

import gaussian_bn as gbn


def main() -> None:
    torch.set_default_dtype(torch.float64)
    # X (0) -> M (1) -> Y (2), affine (per-node offsets c_j)
    m = gbn.GaussianDAG(
        dims=[1, 1, 1],
        edges={(0, 1): [[0.8]], (1, 2): [[0.9]]},
        noise=[[[1.0]], [[0.5]], [[0.4]]],
        mean=[torch.tensor([2.0]), torch.tensor([-1.0]), torch.tensor([0.5])],
    )
    print("marginal means (X, M, Y):", gbn.mean_all(m).tolist())

    # a single observed unit
    e = torch.tensor([3.0, 1.5, 0.7])     # X=3, M=1.5, Y=0.7
    print(f"\nobserved unit: X={e[0]}, M={e[1]}, Y={e[2]}")

    # point intervention do(M = 10): downstream mean of Y shifts
    m_do = gbn.do_hard(m, node=1, value=torch.tensor([10.0]))
    print("do(M=10): E[Y] =", float(gbn.mean_all(m_do)[2]))

    # counterfactual: had X been 5 (it was 3), what would Y have been?
    cf = gbn.counterfactual(m, evidence=[0, 1, 2], evidence_values=e,
                            do={0: torch.tensor([5.0])}, query=[2])
    print(f"\ncounterfactual Y had X been 5 (was 3): {float(cf):.4f}")
    print(f"  change vs observed Y: {float(cf) - float(e[2]):.4f} "
          f"(= a_MY · a_XM · ΔX = 0.9·0.8·2 = {0.9 * 0.8 * 2.0})")

    # sanity: counterfactual with do = factual value recovers the observed Y
    cf0 = gbn.counterfactual(m, [0, 1, 2], e, {0: torch.tensor([3.0])}, [2])
    print(f"  null counterfactual (do X = observed 3): {float(cf0):.4f}  (== observed Y)")


if __name__ == "__main__":
    main()
