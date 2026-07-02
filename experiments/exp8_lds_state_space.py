"""Experiment 8: linear Gaussian state-space model as a running example.

A vector linear dynamical system (first-order VAR observed in noise) exercised
through every framework primitive, and cross-checked against the classical
Kalman filter / Rauch-Tung-Striebel (RTS) smoother where a closed form exists:

    x_n = A x_{n-1} + u_n,   u_n ~ N(0, Q),   x_1 ~ N(0, P0)
    y_n = x_n     + w_n,     w_n ~ N(0, R)

with A = S T (S fixed/known, T the learnable parameter matrix; A tied across all
time steps). As a DAG: states x_0..x_{N-1} are nodes 0..N-1 and observations
y_0..y_{N-1} are nodes N..2N-1.

Primitives exercised (paper Sec. VI):
  E1  inference     : marginal Cov[x_n] vs Kalman prediction recursion;
                      smoother Cov[x_n | y_0..y_{N-1}] vs the RTS recursion.
  E2  conditional independence : Markov  I(x_{n-1}; x_{n+1} | x_n) = 0.
  E3  forward sampling         : sample covariance -> analytic K (1/sqrt(M)).
  E4  estimation               : recover the tied factor T from y only (states hidden).
  E5  identifiability / CRB     : Fisher rank over T; analytic Cramer-Rao bound vs
                                  the empirical scatter of the MLE across sample sizes M.

All numbers are written to results/exp8.json (the source of truth). E5 also
writes pgfplots data (lds_se_vs_m.dat, lds_scatter.dat, lds_ellipse.dat) to the
results directory, or to $LDS_FIGDIR if set (used to regenerate the paper figure).

E5 runs REPS x len(MS) Newton MLE fits (a few minutes); lower REPS for a quick check.

Run:  uv run python experiments/exp8_lds_state_space.py
"""

from __future__ import annotations

import json
import math
import os
import time

import torch

import gaussian_bn as gbn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.environ.get("LDS_FIGDIR", RESULTS)

# ---- model constants (paper Sec. VI-A) ----
d, N = 2, 8
S = torch.tensor([[1.0, 0.5], [0.0, 1.0]])            # fixed, known factor
T_TRUE = torch.tensor([[0.4, -0.6], [0.2, 0.6]])      # the learnable parameter
A = S @ T_TRUE                                         # tied transition
P0, Q, R = torch.eye(d), 0.1 * torch.eye(d), 0.2 * torch.eye(d)

# ---- E5 Monte-Carlo settings (paper) ----
MS = [500, 1000, 2000, 4000, 8000]
REPS = 400
LABELS = ["T11", "T12", "T21", "T22"]


def build_lds(A_mat: torch.Tensor) -> gbn.GaussianDAG:
    dims = [d] * (2 * N)
    edges = {}
    for n in range(1, N):
        edges[(n - 1, n)] = A_mat                      # tied A on every transition
    for n in range(N):
        edges[(n, N + n)] = torch.eye(d)               # y_n = x_n + w_n
    noise = [P0] + [Q] * (N - 1) + [R] * N
    return gbn.GaussianDAG(dims=dims, edges=edges, noise=noise, validate=False)


def kalman_covariances():
    """Classical Kalman prediction / filter / RTS smoother covariances (C = I)."""
    Sig = [P0]                                         # prior (predict) Cov[x_n]
    for n in range(1, N):
        Sig.append(A @ Sig[-1] @ A.T + Q)
    Ppred, Pfilt = [P0], []
    for n in range(N):
        if n > 0:
            Ppred.append(A @ Pfilt[-1] @ A.T + Q)
        Kn = Ppred[n] @ torch.linalg.inv(Ppred[n] + R)
        Pfilt.append((torch.eye(d) - Kn) @ Ppred[n])
    Psm = [None] * N
    Psm[N - 1] = Pfilt[N - 1]
    for n in range(N - 2, -1, -1):
        Cn = Pfilt[n] @ A.T @ torch.linalg.inv(Ppred[n + 1])
        Psm[n] = Pfilt[n] + Cn @ (Psm[n + 1] - Ppred[n + 1]) @ Cn.T
    return Sig, Psm


def run() -> dict:
    m = build_lds(A)
    Kf = gbn.k_full(m)
    obs_nodes = list(range(N, 2 * N))
    oi = m.node_index(obs_nodes)
    out: dict = {"model": {"d": d, "N": N, "spectral_radius_A": float(torch.linalg.eigvals(A).abs().max())}}

    # ---- E1: inference vs Kalman / RTS ----
    Sig, Psm = kalman_covariances()
    e_prior = max(float(torch.linalg.norm(gbn.marginal(m, [n], Kf) - Sig[n])) for n in range(N))
    e_smooth = max(float(torch.linalg.norm(gbn.conditional_covariance(m, [n], obs_nodes, Kf) - Psm[n]))
                   for n in range(N))
    out["E1_inference"] = {
        "max_err_prior_vs_kalman_predict": e_prior,
        "max_err_smoother_vs_rts": e_smooth,
        "lyapunov_gap_last_step": float(torch.linalg.norm(Sig[-1] - Sig[-2])),
    }

    # ---- E2: Markov conditional independence ----
    out["E2_conditional_independence"] = {
        "cmi_markov_on_path": float(gbn.conditional_mutual_information(m, [2], [4], [3], Kf)),
        "cmi_off_path": float(gbn.conditional_mutual_information(m, [2], [4], [5], Kf)),
    }

    # ---- E3: forward sampling -> analytic K ----
    e3 = {}
    for M in (1_000, 10_000, 100_000, 1_000_000):
        X = gbn.sample(m, M, torch.Generator().manual_seed(0))
        rel = float(torch.linalg.norm((X.mH @ X) / M - Kf) / torch.linalg.norm(Kf))
        e3[str(M)] = rel
    out["E3_sampling_relerr_vs_M"] = e3

    # ---- E4: estimate the tied factor T from observations only ----
    Xtr = gbn.sample(m, 200_000, torch.Generator().manual_seed(1))
    patterns = [gbn.ObsPattern(tuple(obs_nodes), Xtr[:, oi])]
    Tp = (0.1 * torch.randn(2, 2, generator=torch.Generator().manual_seed(7))).requires_grad_(True)
    hist = gbn.fit_gradient_custom([Tp], lambda: build_lds(S @ Tp), patterns,
                                   optimizer="adam", lr=0.02, num_iters=1500)
    out["E4_estimation"] = {
        "M_train": 200_000,
        "nll_start": hist[0], "nll_end": hist[-1],
        "err_T": float(torch.linalg.norm(Tp.detach() - T_TRUE)),
        "err_A": float(torch.linalg.norm(S @ Tp.detach() - A)),
    }

    # ---- E5: Fisher over T, and CRB vs empirical MLE scatter ----
    def K_of_eta(eta):
        return gbn.k_full(build_lds(S @ eta.reshape(2, 2)))[oi][:, oi]

    eta0 = T_TRUE.reshape(-1).clone()
    G, w, U = gbn.fisher_metric(K_of_eta, eta0)
    Ginv = torch.linalg.inv(G)
    crb_diag = torch.diag(Ginv)

    def _nll(eta, S_obs):
        Lc = torch.linalg.cholesky(gbn.k_full(build_lds(S @ eta.reshape(2, 2)))[oi][:, oi])
        return 2 * torch.log(torch.diagonal(Lc)).sum() + torch.trace(torch.cholesky_solve(S_obs, Lc))

    def fit_T(S_obs):
        # Newton on the 4-parameter Gaussian NLL: quadratic convergence to the
        # exact MLE in a few steps (fast, and no under-convergence shrinkage).
        e = eta0.clone()
        for _ in range(10):
            g = torch.autograd.functional.jacobian(lambda x: _nll(x, S_obs), e)
            H = torch.autograd.functional.hessian(lambda x: _nll(x, S_obs), e)
            e = e - torch.linalg.solve(H, g)
        return e.detach()

    t0 = time.time()
    se_rows, scatter2000 = {}, None
    for mi, M in enumerate(MS):
        E = torch.stack([
            fit_T((lambda Y: (Y.mH @ Y) / M)(gbn.sample(m, M, torch.Generator().manual_seed(10000 + mi * 1000 + r))[:, oi]))
            for r in range(REPS)
        ])
        emp_se = E.std(dim=0, unbiased=True)
        crb_se = torch.sqrt(crb_diag / M)
        se_rows[str(M)] = {"crb": [float(x) for x in crb_se], "emp": [float(x) for x in emp_se]}
        if M == 2000:
            scatter2000 = E
    out["E5_reliability"] = {
        "fisher_eigs": [float(x) for x in w],
        "fisher_rank": int((w > 1e-8 * w.max()).sum()),
        "q": int(G.shape[0]),
        "labels": LABELS,
        "REPS": REPS,
        "se_vs_M": se_rows,
        "sweep_seconds": time.time() - t0,
    }

    _write_figdata(se_rows, scatter2000, Ginv)
    return out


def _write_figdata(se_rows, scatter2000, Ginv) -> None:
    """pgfplots data for the paper figure (Sec. VI-E)."""
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, "lds_se_vs_m.dat"), "w") as f:
        f.write("M " + " ".join(f"crb{l} emp{l}" for l in LABELS) + "\n")
        for M in MS:
            row = se_rows[str(M)]
            f.write(f"{M} " + " ".join(f"{row['crb'][k]:.6e} {row['emp'][k]:.6e}" for k in range(4)) + "\n")
    with open(os.path.join(FIGDIR, "lds_scatter.dat"), "w") as f:
        f.write("T11 T21\n")
        for r in range(scatter2000.shape[0]):
            f.write(f"{float(scatter2000[r, 0]):.6e} {float(scatter2000[r, 2]):.6e}\n")
    cov2 = Ginv[[0, 2]][:, [0, 2]] / 2000.0
    L2 = torch.linalg.cholesky(cov2)
    scale = math.sqrt(5.9915)                          # chi2_{0.95, 2}
    cx, cy = float(T_TRUE[0, 0]), float(T_TRUE[1, 0])
    with open(os.path.join(FIGDIR, "lds_ellipse.dat"), "w") as f:
        f.write("x y\n")
        for j in range(129):
            th = 2 * math.pi * j / 128
            v = scale * (L2 @ torch.tensor([math.cos(th), math.sin(th)]))
            f.write(f"{cx + float(v[0]):.6e} {cy + float(v[1]):.6e}\n")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = run()
    with open(os.path.join(RESULTS, "exp8.json"), "w") as f:
        json.dump(out, f, indent=2)

    e1 = out["E1_inference"]
    print(f"E1  prior vs Kalman-predict : max err {e1['max_err_prior_vs_kalman_predict']:.2e}")
    print(f"E1  smoother vs RTS         : max err {e1['max_err_smoother_vs_rts']:.2e}")
    e2 = out["E2_conditional_independence"]
    print(f"E2  Markov CMI on-path/off  : {e2['cmi_markov_on_path']:.2e} / {e2['cmi_off_path']:.4f}")
    print(f"E3  sampling rel.err @1e6    : {out['E3_sampling_relerr_vs_M']['1000000']:.2e}")
    print(f"E4  tied-T recovery err      : {out['E4_estimation']['err_T']:.2e}")
    e5 = out["E5_reliability"]
    print(f"E5  Fisher rank {e5['fisher_rank']}/{e5['q']}, eigs {[round(x,2) for x in e5['fisher_eigs']]}")
    for M in MS:
        row = e5["se_vs_M"][str(M)]
        print(f"E5  M={M:5d}  " + "  ".join(
            f"{LABELS[k]} crb={row['crb'][k]:.4f} emp={row['emp'][k]:.4f}" for k in range(4)))
    print(f"\nwrote results/exp8.json and lds_*.dat to {FIGDIR}")


if __name__ == "__main__":
    main()
