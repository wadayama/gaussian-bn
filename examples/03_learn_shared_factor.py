"""Example 3: learn a shared edge factor (parameter sharing) from data.

A relay node 0 broadcasts the same processing matrix `F` to two children through
fixed channels: edge `0->1` carries `H1 @ F` and edge `0->2` carries `H2 @ F`.
The single shared factor `F` (and the node noises) is learned from data with
`fit_gradient_custom`, which back-propagates the marginal likelihood through any
construction of the edge matrices.

Run:  uv run python examples/03_learn_shared_factor.py
"""

from __future__ import annotations

import torch

import gaussian_bn as gbn


def main() -> None:
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(10)
    H1 = torch.randn(2, 2, generator=g)
    H2 = torch.randn(2, 2, generator=g)
    F_true = torch.tensor([[0.6, -0.3], [0.2, 0.5]])

    # ground-truth model with the SHARED factor F baked into both edges
    true = gbn.GaussianDAG(
        dims=[2, 2, 2],
        edges={(0, 1): H1 @ F_true, (0, 2): H2 @ F_true},
        noise=[torch.eye(2), 0.3 * torch.eye(2), 0.3 * torch.eye(2)],
    )
    X = gbn.sample(true, 100_000, torch.Generator().manual_seed(11))
    patterns = [gbn.ObsPattern((0, 1, 2), X)]  # full observation

    # learnable parameters: the shared factor F and per-node log-noise diagonals
    F = (0.1 * torch.randn(2, 2, generator=torch.Generator().manual_seed(12))).requires_grad_(True)
    log_noise = [torch.zeros(2, requires_grad=True) for _ in range(3)]

    def build_model() -> gbn.GaussianDAG:
        return gbn.GaussianDAG(
            dims=[2, 2, 2],
            edges={(0, 1): H1 @ F, (0, 2): H2 @ F},          # F shared across both edges
            noise=[torch.diag(torch.exp(s)) for s in log_noise],
            validate=False,
        )

    history = gbn.fit_gradient_custom(
        [F, *log_noise], build_model, patterns,
        optimizer="adam", lr=0.02, num_iters=1500,
    )

    err = float(torch.linalg.norm(F.detach() - F_true))
    print(f"NLL: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"shared-factor recovery: ||F_hat - F_true|| = {err:.5f}")
    print("F_true =\n", F_true.numpy())
    print("F_hat  =\n", F.detach().numpy().round(4))

    # the learned model's reliability is available the same way as any other fit
    fitted = build_model()
    print("\nlearned edge A[0->1] = H1 @ F_hat =\n", fitted.edges[(0, 1)].detach().numpy().round(4))


if __name__ == "__main__":
    main()
