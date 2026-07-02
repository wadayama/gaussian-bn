"""Experiment 10: skip-connection extension of the LDS running example
(paper Sec. VI-F).

The state-space model of exp8 gains a tied skip connection,

    x_n = A x_{n-1} + C x_{n-2} + u_n,      y_n = x_n + w_n,

with A = S T tied as in exp8 (T the learnable factor) and C a known skip
matrix shared across all steps.  Every state x_n with n >= 3 merges two
correlated parents, so the parent cross-covariances of the K-recursion are
essential and the first-order Kalman recursions no longer apply to the graph
as given; classically one re-derives the model in companion form (stacking
(x_n, x_{n-1}) into a first-order state).  The framework needs no such
reformulation and is validated against it:

  S1  companion check : Cov[x_n] from the chart vs the hand-augmented
                        companion-form prediction recursion (machine
                        precision).
  S2  CMI vs structure: the first-order Markov CMI I(x_{n-1};x_{n+1}|x_n),
                        which vanished on the chain (exp8 E2), is now positive
                        (the skip edge keeps a path open), while the
                        second-order CMI I(x_{n-1};x_{n+2}|x_n,x_{n+1}) = 0.
  S3  estimation      : T recovered from observations y only (states hidden),
                        random init, exactly as exp8 E4, unchanged code.
  S4  reliability     : Fisher over T full rank; CRB standard errors match
                        the empirical MLE scatter (Newton fits, as exp8 E5).

Writes results/exp10.json (the source of truth; all reported digits come from
this run) and pgfplots data for the paper figure (skip_scatter.dat,
skip_ellipse.dat) to the results directory, or to $LDS_FIGDIR if set.

Run:  uv run python experiments/exp10_lds_skip.py
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

# ---- model: exp8 LDS + tied skip connection ----
d, N = 2, 8
S = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
T_TRUE = torch.tensor([[0.4, -0.6], [0.2, 0.6]])
A = S @ T_TRUE
C = torch.tensor([[0.15, 0.0], [0.05, 0.1]])       # known tied skip matrix
P0, Q, R = torch.eye(d), 0.1 * torch.eye(d), 0.2 * torch.eye(d)

# ---- Monte-Carlo settings (paper) ----
M_FIT = 200_000
M_MC, REPS = 2000, 400
LABELS = ["T11", "T12", "T21", "T22"]


def build_skip(A_mat: torch.Tensor) -> gbn.GaussianDAG:
    dims = [d] * (2 * N)
    edges = {}
    for n in range(1, N):
        edges[(n - 1, n)] = A_mat                   # tied A on every transition
    for n in range(2, N):
        edges[(n - 2, n)] = C                       # tied skip on every step
    for n in range(N):
        edges[(n, N + n)] = torch.eye(d)            # y_n = x_n + w_n
    noise = [P0] + [Q] * (N - 1) + [R] * N
    return gbn.GaussianDAG(dims=dims, edges=edges, noise=noise, validate=False)


def companion_prior_covariances():
    """Prediction recursion on the hand-augmented companion form
    z_n = (x_n, x_{n-1}):  z_n = F z_{n-1} + (u_n, 0)."""
    F = torch.zeros(2 * d, 2 * d)
    F[:d, :d], F[:d, d:], F[d:, :d] = A, C, torch.eye(d)
    Qz = torch.zeros(2 * d, 2 * d)
    Qz[:d, :d] = Q
    # z_2 = (x_2, x_1) with x_1 ~ P0, x_2 = A x_1 + u_2
    Sz = torch.zeros(2 * d, 2 * d)
    Sz[:d, :d] = A @ P0 @ A.T + Q
    Sz[:d, d:] = A @ P0
    Sz[d:, :d] = P0 @ A.T
    Sz[d:, d:] = P0
    covs = {0: P0, 1: Sz[:d, :d].clone()}
    for n in range(2, N):
        Sz = F @ Sz @ F.T + Qz
        covs[n] = Sz[:d, :d].clone()
    return covs, float(torch.linalg.eigvals(F).abs().max())


def run() -> dict:
    m = build_skip(A)
    Kf = gbn.k_full(m)
    obs_nodes = list(range(N, 2 * N))
    oi = m.node_index(obs_nodes)
    covs_comp, rho_F = companion_prior_covariances()
    out: dict = {"model": {"d": d, "N": N,
                           "spectral_radius_companion": rho_F}}

    # ---- S1: framework marginals vs companion-form prediction recursion ----
    e_comp = max(float(torch.linalg.norm(gbn.marginal(m, [n], Kf) - covs_comp[n]))
                 for n in range(N))
    # parent cross-covariance at a representative merging node x_5 (0-based 4)
    K_parents = Kf[d * 3:d * 4, d * 2:d * 3]        # K_{x4, x3} (0-based)
    out["S1_companion"] = {
        "max_err_prior_vs_companion_kalman": e_comp,
        "norm_parent_cross_cov_x4x3": float(torch.linalg.norm(K_parents)),
    }

    # ---- S2: CMI tracks the new structure ----
    out["S2_conditional_independence"] = {
        "cmi_first_order_markov": float(gbn.conditional_mutual_information(m, [1], [3], [2], Kf)),
        "cmi_second_order_markov": float(gbn.conditional_mutual_information(m, [1], [4], [2, 3], Kf)),
    }

    # ---- S3: estimate the tied factor T from observations only ----
    Xtr = gbn.sample(m, M_FIT, torch.Generator().manual_seed(1))
    patterns = [gbn.ObsPattern(tuple(obs_nodes), Xtr[:, oi])]
    Tp = (0.1 * torch.randn(2, 2, generator=torch.Generator().manual_seed(7))).requires_grad_(True)
    hist = gbn.fit_gradient_custom([Tp], lambda: build_skip(S @ Tp), patterns,
                                   optimizer="adam", lr=0.02, num_iters=1500)
    out["S3_estimation"] = {
        "M_train": M_FIT,
        "nll_start": hist[0], "nll_end": hist[-1],
        "err_T": float(torch.linalg.norm(Tp.detach() - T_TRUE)),
        "err_A": float(torch.linalg.norm(S @ Tp.detach() - A)),
    }

    # ---- S4: Fisher over T, CRB vs empirical MLE scatter, for two
    #          observation patterns: all sensors vs the odd-indexed half ----
    eta0 = T_TRUE.reshape(-1).clone()

    def crb_mc(obs_sub: list, seed: int) -> tuple:
        oi_sub = m.node_index(obs_sub)

        def K_of_eta(eta):
            return gbn.k_full(build_skip(S @ eta.reshape(2, 2)))[oi_sub][:, oi_sub]

        G, w, U = gbn.fisher_metric(K_of_eta, eta0)
        rank = int((w > 1e-10 * w.max()).sum())
        Ginv = torch.linalg.inv(G)

        def _nll(eta, S_obs):
            Lc = torch.linalg.cholesky(K_of_eta(eta))
            return 2 * torch.log(torch.diagonal(Lc)).sum() + torch.trace(torch.cholesky_solve(S_obs, Lc))

        def fit_newton(S_obs):
            e = eta0.clone()
            for _ in range(10):
                g = torch.autograd.functional.jacobian(lambda x: _nll(x, S_obs), e)
                H = torch.autograd.functional.hessian(lambda x: _nll(x, S_obs), e)
                e = e - torch.linalg.solve(H, g)
            return e.detach()

        t0 = time.time()
        gen = torch.Generator().manual_seed(seed)
        ests = []
        for _ in range(REPS):
            Xs = gbn.sample(m, M_MC, gen)[:, oi_sub]
            ests.append(fit_newton((Xs.T @ Xs) / M_MC))
        E = torch.stack(ests)
        stats = {
            "observed_y": [n - N + 1 for n in obs_sub],    # 1-based sensor indices
            "fisher_rank": rank, "q": int(G.shape[0]),
            "fisher_eigenvalues": [float(v) for v in w],
            "M": M_MC, "reps": REPS,
            "crb_se": [float(v) for v in torch.sqrt(torch.diag(Ginv) / M_MC)],
            "empirical_se": [float(v) for v in E.std(dim=0, unbiased=True)],
            "mean_bias": [float(v) for v in (E.mean(dim=0) - eta0)],
            "seconds": time.time() - t0,
        }
        return stats, E, Ginv

    odd_nodes = [N + n for n in range(0, N, 2)]            # y_1, y_3, y_5, y_7
    out["S4_crb"], E_full, Ginv_full = crb_mc(obs_nodes, seed=2)
    out["S4_crb_partial"], E_part, Ginv_part = crb_mc(odd_nodes, seed=3)
    _write_figdata("skip", E_full, Ginv_full)
    _write_figdata("skip_partial", E_part, Ginv_part)
    return out


def _write_figdata(prefix: str, E, Ginv) -> None:
    """pgfplots data for the paper figure (Sec. VI-F): the M=2000 estimates
    (T11_hat, T21_hat) and the analytic 95% Cramer-Rao ellipse, in the same
    format as exp8's lds_scatter.dat / lds_ellipse.dat."""
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, f"{prefix}_scatter.dat"), "w") as f:
        f.write("T11 T21\n")
        for r in range(E.shape[0]):
            f.write(f"{float(E[r, 0]):.6e} {float(E[r, 2]):.6e}\n")
    cov2 = Ginv[[0, 2]][:, [0, 2]] / M_MC
    L2 = torch.linalg.cholesky(cov2)
    scale = math.sqrt(5.9915)                          # chi2_{0.95, 2}
    cx, cy = float(T_TRUE[0, 0]), float(T_TRUE[1, 0])
    with open(os.path.join(FIGDIR, f"{prefix}_ellipse.dat"), "w") as f:
        f.write("x y\n")
        for j in range(129):
            th = 2 * math.pi * j / 128
            v = scale * (L2 @ torch.tensor([math.cos(th), math.sin(th)]))
            f.write(f"{cx + float(v[0]):.6e} {cy + float(v[1]):.6e}\n")


if __name__ == "__main__":
    res = run()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "exp10.json"), "w") as f:
        json.dump(res, f, indent=1)

    s1, s2, s3, s4 = (res["S1_companion"], res["S2_conditional_independence"],
                      res["S3_estimation"], res["S4_crb"])
    print(f"model  rho(companion F)        : {res['model']['spectral_radius_companion']:.3f}")
    print(f"S1  prior vs companion Kalman  : max err {s1['max_err_prior_vs_companion_kalman']:.2e}")
    print(f"S1  ||K_parents(x4,x3)||       : {s1['norm_parent_cross_cov_x4x3']:.4f}")
    print(f"S2  I(x2;x4|x3)  (1st order)   : {s2['cmi_first_order_markov']:.4f}")
    print(f"S2  I(x2;x5|x3,x4) (2nd order) : {s2['cmi_second_order_markov']:.3e}")
    print(f"S3  ||T_hat - T||              : {s3['err_T']:.3e}  (NLL {s3['nll_start']:.2f} -> {s3['nll_end']:.2f})")
    print(f"S4  Fisher rank                : {s4['fisher_rank']}/{s4['q']}  eig {['%.2f' % v for v in s4['fisher_eigenvalues']]}")
    print(f"S4  CRB se  (M={s4['M']})          : {['%.4f' % v for v in s4['crb_se']]}")
    print(f"S4  emp se  ({s4['reps']} trials)     : {['%.4f' % v for v in s4['empirical_se']]}")
    print(f"S4  MC time                    : {s4['seconds']:.1f} s")
    sp = res["S4_crb_partial"]
    print(f"S4p observed y                 : {sp['observed_y']}  rank {sp['fisher_rank']}/{sp['q']}")
    print(f"S4p CRB se  (M={sp['M']})          : {['%.4f' % v for v in sp['crb_se']]}")
    print(f"S4p emp se  ({sp['reps']} trials)     : {['%.4f' % v for v in sp['empirical_se']]}")
    print(f"S4p MC time                    : {sp['seconds']:.1f} s")
