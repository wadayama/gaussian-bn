"""Experiment 9: blind self-calibration of a sensor-fusion network.

A non-chain (fusion) topology: two hidden latent roots, a source S in R^2 and an
interference J in R, feed K = 8 scalar sensors

    y_k = H_k S + g_k J + w_k ,   w_k ~ N(0, sigma^2),

so every sensor is a MERGING node of the two roots (the parent cross-covariances
of the K-recursion self-block are essential; a chain recursion does not apply).

Sensors 1..3 are UNCALIBRATED: their combined loading rows
lambda_k = (H_k, g_k) in R^3 are the unknown parameters (q = 9). Sensors 4..8
are CALIBRATED references with known loadings. The latents are never observed,
and their covariance is fixed to the identity, so the blind problem carries the
classical factor-analysis rotation gauge: any Q in O(3) with lambda_k -> lambda_k Q^T
(all k) leaves the observed covariance invariant.

Sweep: observe the three uncalibrated sensors together with every subset of the
five references (2^5 = 32 subsets, exhaustive). Theory predicts the Fisher rank
deficit as the dimension of the stabilizer of the observed reference rows in
O(3):

    m = 0 references -> deficit 3   (full so(3) gauge)
    m = 1            -> deficit 1   (rotations about the one anchored axis)
    m >= 2 (generic) -> deficit 0   (identifiable)

Checks:
  C1  the measured rank deficit matches the {3, 1, 0} staircase for all subsets;
  C2  past the threshold, the Cramer-Rao standard errors are finite, shrink as
      references are added, and SPREAD strongly across subsets of equal size
      (placement quality);
  C2b for the best and the worst 2-reference placement, the weakest Fisher
      eigendirection is a residual rotation of the unknown loading rows (its
      overlap with the tangent space of the O(3) gauge orbit is ~1), and the
      placement quality is set by how strongly the data excite that rotation
      (the smallest Fisher eigenvalue);
  C3  Monte-Carlo MLE scatter matches the analytic CRB for both the best and the
      worst identifiable 2-reference placement (the Fisher metric ranks
      placements correctly).

All numbers are written to results/exp9.json (the source of truth); pgfplots
data for the paper figure goes to results/calib_*.dat (or $CALIB_FIGDIR).

Run:  uv run python experiments/exp9_sensor_calibration.py
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time

import torch

import gaussian_bn as gbn

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.environ.get("CALIB_FIGDIR", RESULTS)

# ---- model constants ----
r = 3                    # latent dimension: source (2) + interference (1)
K = 8                    # sensors
N_UNC = 3                # uncalibrated sensors (nodes 2, 3, 4)
SIGMA2 = 0.1             # sensor noise variance (known)
UNC = [2, 3, 4]
REFS = [5, 6, 7, 8, 9]

# Ground-truth combined loading rows lambda_k in R^3 (seeded, generic).
_g = torch.Generator().manual_seed(42)
LAM_TRUE = torch.randn((K, r), generator=_g)

# ---- Monte-Carlo settings (C3) ----
M_MC = 2000              # snapshots per trial
REPS = 200               # trials per configuration


def build_model(lam_unc: torch.Tensor) -> gbn.GaussianDAG:
    """Fusion DAG: node 0 = S (dim 2), node 1 = J (dim 1), nodes 2..9 = sensors."""
    lam = LAM_TRUE.clone().to(lam_unc.dtype)
    lam = torch.cat([lam[:N_UNC] * 0 + lam_unc.reshape(N_UNC, r), lam[N_UNC:]], dim=0)
    dims = [2, 1] + [1] * K
    edges = {}
    for k in range(K):
        edges[(0, 2 + k)] = lam[k, :2].reshape(1, 2)     # H_k
        edges[(1, 2 + k)] = lam[k, 2:].reshape(1, 1)     # g_k
    noise = [torch.eye(2), torch.eye(1)] + [SIGMA2 * torch.eye(1)] * K
    return gbn.GaussianDAG(dims=dims, edges=edges, noise=noise, validate=False)


ETA0 = LAM_TRUE[:N_UNC].reshape(-1).clone()              # q = 9 true parameters


def K_of_eta_factory(observed):
    def K_of_eta(eta: torch.Tensor) -> torch.Tensor:
        mm = build_model(eta)
        return gbn.k_full(mm)[mm.node_index(observed)][:, mm.node_index(observed)]
    return K_of_eta


def nll(eta: torch.Tensor, S_obs: torch.Tensor, observed) -> torch.Tensor:
    K_OO = K_of_eta_factory(observed)(eta)
    L = torch.linalg.cholesky(K_OO)
    return 2 * torch.log(torch.diagonal(L)).sum() + torch.trace(torch.cholesky_solve(S_obs, L))


def fit_newton(S_obs: torch.Tensor, observed) -> torch.Tensor:
    e = ETA0.clone()
    for _ in range(10):
        g = torch.autograd.functional.jacobian(lambda x: nll(x, S_obs, observed), e)
        H = torch.autograd.functional.hessian(lambda x: nll(x, S_obs, observed), e)
        e = e - torch.linalg.solve(H, g)
    return e.detach()


def gauge_mode_analysis(best: dict, worst: dict) -> dict:
    """C2b: the weakest Fisher eigendirection vs the O(3) gauge-orbit tangent.

    The gauge orbit acts on the stacked uncalibrated rows by
    lam_unc -> lam_unc Q^T; its tangent space at the truth is spanned by
    lam_unc J_a^T for the three so(3) generators J_a. For each extreme m = 2
    placement this records the Fisher eigenvalues, the overlap of the weakest
    eigenvector with that tangent space, and (for the record) the angle between
    the two known reference rows.
    """
    gens = []
    for i, j in ((1, 2), (2, 0), (0, 1)):
        J = torch.zeros(3, 3)
        J[i, j], J[j, i] = -1.0, 1.0
        gens.append(J)
    T = torch.stack([(LAM_TRUE[:N_UNC] @ J.T).reshape(-1) for J in gens])  # 3 x q
    Q, _ = torch.linalg.qr(T.T)                       # orthonormal basis, q x 3
    out = {}
    for tag, cfg in (("best", best), ("worst", worst)):
        observed = UNC + cfg["refs"]
        G, w, U = gbn.fisher_metric(K_of_eta_factory(observed), ETA0)
        overlap = float(torch.linalg.norm(Q.T @ U[:, 0]))
        r1, r2 = (LAM_TRUE[k - 2] for k in cfg["refs"])
        cosang = float(torch.dot(r1, r2) / (r1.norm() * r2.norm()))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        out[tag] = {
            "refs": cfg["refs"],
            "fisher_eigs": [float(x) for x in w],
            "min_fisher_eig": float(w[0]),
            "weakest_dir_gauge_overlap": overlap,
            "ref_row_angle_deg": min(ang, 180.0 - ang),
        }
    return out


def run() -> dict:
    out: dict = {"q": int(ETA0.numel()), "sigma2": SIGMA2,
                 "expected_deficit": {"m=0": 3, "m=1": 1, "m>=2": 0}}
    m_true = build_model(ETA0)

    # ---- C1 + C2: exhaustive sweep over reference subsets ----
    sweep = []
    for msize in range(len(REFS) + 1):
        for sel in itertools.combinations(REFS, msize):
            observed = UNC + list(sel)
            G, w, U = gbn.fisher_metric(K_of_eta_factory(observed), ETA0)
            q = w.numel()
            rank = int((w > 1e-8 * float(w.max())).sum())
            entry = {"refs": list(sel), "m": msize, "rank": rank, "deficit": q - rank}
            if rank == q:
                crb_se = torch.sqrt(torch.diag(torch.linalg.inv(G)) / M_MC)
                entry["mean_crb_se"] = float(crb_se.mean())
            sweep.append(entry)
    out["sweep"] = sweep

    deficits = {}
    for e in sweep:
        deficits.setdefault(e["m"], set()).add(e["deficit"])
    out["deficit_by_m"] = {str(m): sorted(v) for m, v in sorted(deficits.items())}

    # ---- C3: Monte-Carlo validation of best / worst 2-reference placement ----
    pairs = [e for e in sweep if e["m"] == 2 and "mean_crb_se" in e]
    best = min(pairs, key=lambda e: e["mean_crb_se"])
    worst = max(pairs, key=lambda e: e["mean_crb_se"])
    out["gauge_mode"] = gauge_mode_analysis(best, worst)
    out["mc"] = {}
    t0 = time.time()
    for tag, cfg in (("best", best), ("worst", worst)):
        observed = UNC + cfg["refs"]
        oi = m_true.node_index(observed)
        ests = []
        for rep in range(REPS):
            X = gbn.sample(m_true, M_MC, torch.Generator().manual_seed(50000 + rep))
            Y = X[:, oi]
            ests.append(fit_newton((Y.mH @ Y) / M_MC, observed))
        E = torch.stack(ests)
        emp_se = E.std(dim=0, unbiased=True)
        G, w, U = gbn.fisher_metric(K_of_eta_factory(observed), ETA0)
        crb_se = torch.sqrt(torch.diag(torch.linalg.inv(G)) / M_MC)
        out["mc"][tag] = {
            "refs": cfg["refs"],
            "mean_crb_se": float(crb_se.mean()),
            "mean_emp_se": float(emp_se.mean()),
            "crb_se": [float(x) for x in crb_se],
            "emp_se": [float(x) for x in emp_se],
        }
    out["mc"]["REPS"] = REPS
    out["mc"]["M"] = M_MC
    out["mc_seconds"] = time.time() - t0

    _write_figdata(sweep, out["mc"])
    return out


def _write_figdata(sweep, mc) -> None:
    """pgfplots data: per-subset mean CRB SE vs m, plus the two MC points."""
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, "calib_sweep.dat"), "w") as f:
        f.write("m se\n")
        for e in sweep:
            if "mean_crb_se" in e:
                f.write(f"{e['m']} {e['mean_crb_se']:.6e}\n")
    with open(os.path.join(FIGDIR, "calib_mc.dat"), "w") as f:
        f.write("m se\n")
        for tag in ("best", "worst"):
            f.write(f"2 {mc[tag]['mean_emp_se']:.6e}\n")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = run()
    with open(os.path.join(RESULTS, "exp9.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"q = {out['q']} uncalibrated loading parameters, sigma^2 = {out['sigma2']}")
    print("C1  rank deficit by #references m (measured, over ALL subsets):")
    for m, ds in out["deficit_by_m"].items():
        print(f"    m={m}: deficit(s) {ds}")
    print("    expected staircase: m=0 -> 3, m=1 -> 1, m>=2 -> 0")
    pairs = [e for e in out["sweep"] if e["m"] == 2 and "mean_crb_se" in e]
    ses = sorted(e["mean_crb_se"] for e in pairs)
    print(f"C2  m=2 placements: mean CRB SE ranges {ses[0]:.4f} .. {ses[-1]:.4f} "
          f"({ses[-1]/ses[0]:.1f}x spread across placements)")
    for tag in ("best", "worst"):
        gm = out["gauge_mode"][tag]
        print(f"C2b {tag:5s} refs={gm['refs']}  min Fisher eig={gm['min_fisher_eig']:.4f}  "
              f"weakest-dir gauge overlap={gm['weakest_dir_gauge_overlap']:.4f}  "
              f"ref-row angle={gm['ref_row_angle_deg']:.1f} deg")
    for tag in ("best", "worst"):
        mc = out["mc"][tag]
        print(f"C3  {tag:5s} refs={mc['refs']}  mean SE: crb={mc['mean_crb_se']:.4f} "
              f"emp={mc['mean_emp_se']:.4f}  ratio={mc['mean_emp_se']/mc['mean_crb_se']:.3f}")
    print(f"\nwrote results/exp9.json and calib_*.dat to {FIGDIR}")


if __name__ == "__main__":
    main()
