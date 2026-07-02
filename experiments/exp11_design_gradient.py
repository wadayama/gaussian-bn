"""Experiment 11: continuous D-optimal design by nested automatic differentiation.

Companion to experiment 9 (same fusion model, same seed): three uncalibrated
sensors, latents hidden, and the WORST 2-reference placement of exp9 as the
starting point (reference loading rows only 8.9 degrees apart, so the rotation
gauge is poorly pinned and the CRB is large).

Here the placement is made CONTINUOUS: one of the two co-observed references
may re-aim its loading row (fixed norm, direction free), and we ascend the
D-optimality objective

    phi(p) = logdet G(row = rho * p / |p|)

by Adam, where G is the q = 9 Fisher metric of the uncalibrated loadings. The
gradient d phi / d p is a NESTED automatic derivative: G itself contains the
inner Jacobian dK_OO/d eta (forward-mode), and reverse-mode back-propagates
through it (fisher_metric_differentiable).

Checks:
  C1  at the starting point, the nested-AD gradient of phi matches central
      finite differences;
  C2  the objective increases along the ascent, and the mean CRB standard
      error (at M = 2000 snapshots) drops from the exp9 worst-placement value
      toward, and past, the best exhaustive m = 2 placement;
  C3  the optimizer's geometry is interpretable: the re-aimed row rotates away
      from near-collinearity with the fixed reference row.

All numbers are written to results/exp11.json (the source of truth).

Run:  uv run python experiments/exp11_design_gradient.py
"""

from __future__ import annotations

import json
import math
import os

import torch

import gaussian_bn as gbn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---- same fusion model as experiment 9 (same seed) ----
r = 3
K = 8
N_UNC = 3
SIGMA2 = 0.1
_g = torch.Generator().manual_seed(42)
LAM_TRUE = torch.randn((K, r), generator=_g)
ETA0 = LAM_TRUE[:N_UNC].reshape(-1).clone()

FIXED_REF = 5          # node index of the fixed reference   (sensor 4)
FREE_REF = 9           # node index of the re-aimable one    (sensor 8)
OBSERVED = [2, 3, 4, FIXED_REF, FREE_REF]
M_SNAP = 2000          # snapshot count for the reported CRB scale
LR, ITERS = 0.05, 300


def _build(lam_unc: torch.Tensor, row_free: torch.Tensor) -> gbn.GaussianDAG:
    """exp9's fusion DAG with free uncalibrated loadings ``lam_unc`` and the
    FREE_REF reference row replaced by the design row ``row_free``."""
    rows = [lam_unc.reshape(N_UNC, r)[k] if k < N_UNC
            else (row_free if 2 + k == FREE_REF else LAM_TRUE[k])
            for k in range(K)]
    dims = [2, 1] + [1] * K
    edges = {}
    for k, lam_k in enumerate(rows):
        edges[(0, 2 + k)] = lam_k[:2].reshape(1, 2)
        edges[(1, 2 + k)] = lam_k[2:].reshape(1, 1)
    noise = [torch.eye(2), torch.eye(1)] + [SIGMA2 * torch.eye(1)] * K
    return gbn.GaussianDAG(dims=dims, edges=edges, noise=noise, validate=False)


def G_of_row(row_free: torch.Tensor) -> torch.Tensor:
    def K_of_eta(eta: torch.Tensor) -> torch.Tensor:
        m = _build(eta, row_free)
        oi = m.node_index(OBSERVED)
        return gbn.k_full(m)[oi][:, oi]

    return gbn.fisher_metric_differentiable(K_of_eta, ETA0)


ROW0 = LAM_TRUE[FREE_REF - 2].clone()
RHO = float(ROW0.norm())                       # power (row norm) held fixed


def phi(p: torch.Tensor) -> torch.Tensor:
    return torch.logdet(G_of_row(RHO * p / p.norm()))


def mean_crb_se(p: torch.Tensor) -> float:
    G = G_of_row(RHO * p.detach() / p.detach().norm())
    return float(torch.sqrt(torch.diag(torch.linalg.inv(G)) / M_SNAP).mean())


def angle_deg(u: torch.Tensor, v: torch.Tensor) -> float:
    c = abs(float(u @ v / (u.norm() * v.norm())))
    return math.degrees(math.acos(min(c, 1.0)))


def run() -> dict:
    out: dict = {"observed_nodes": OBSERVED, "M": M_SNAP, "lr": LR, "iters": ITERS}

    # ---- C1: nested-AD gradient vs finite differences at the starting point ----
    p = ROW0.clone().requires_grad_(True)
    (g_ad,) = torch.autograd.grad(phi(p), p)
    h = 1e-6
    fd = torch.zeros(3)
    with torch.no_grad():
        for i in range(3):
            e = torch.zeros(3); e[i] = h
            fd[i] = (phi(ROW0 + e) - phi(ROW0 - e)) / (2 * h)
    rel = float(torch.linalg.norm(g_ad - fd) / torch.linalg.norm(fd))
    out["C1_grad_check"] = {"rel_err_vs_fd": rel,
                            "grad_norm": float(g_ad.norm())}

    # ---- C2: Adam ascent on logdet G ----
    p = ROW0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=LR)
    traj = []
    for t in range(ITERS):
        opt.zero_grad()
        loss = -phi(p)
        loss.backward()
        opt.step()
        if t % 25 == 0 or t == ITERS - 1:
            traj.append({"iter": t, "logdetG": -float(loss.detach()),
                         "mean_crb_se": mean_crb_se(p)})
    out["C2_trajectory"] = traj
    out["C2_summary"] = {
        "logdetG_init": traj[0]["logdetG"], "logdetG_final": traj[-1]["logdetG"],
        "mean_crb_se_init": mean_crb_se(ROW0),
        "mean_crb_se_final": mean_crb_se(p),
        "exp9_worst_pair_se": 0.0372,   # exp9.json, refs [5, 9]
        "exp9_best_pair_se": 0.0150,    # exp9.json, refs [5, 7]
    }

    # ---- C3: geometry of the optimized row ----
    row_opt = (RHO * p.detach() / p.detach().norm())
    out["C3_geometry"] = {
        "angle_init_deg": angle_deg(ROW0, LAM_TRUE[FIXED_REF - 2]),
        "angle_final_deg": angle_deg(row_opt, LAM_TRUE[FIXED_REF - 2]),
        "row_init": [float(x) for x in ROW0],
        "row_final": [float(x) for x in row_opt],
    }
    return out


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = run()
    with open(os.path.join(RESULTS, "exp11.json"), "w") as f:
        json.dump(out, f, indent=2)

    c1, c2, c3 = out["C1_grad_check"], out["C2_summary"], out["C3_geometry"]
    print(f"C1  nested-AD gradient vs finite differences: rel err {c1['rel_err_vs_fd']:.2e}")
    print(f"C2  logdet G: {c2['logdetG_init']:.3f} -> {c2['logdetG_final']:.3f}")
    print(f"C2  mean CRB SE: {c2['mean_crb_se_init']:.4f} -> {c2['mean_crb_se_final']:.4f}"
          f"  (exp9 worst {c2['exp9_worst_pair_se']}, best discrete {c2['exp9_best_pair_se']})")
    print(f"C3  angle to fixed reference: {c3['angle_init_deg']:.1f} deg -> "
          f"{c3['angle_final_deg']:.1f} deg")
    print("\nwrote results/exp11.json")


if __name__ == "__main__":
    main()
