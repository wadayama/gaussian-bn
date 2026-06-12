"""Experiment 7: accuracy-declared SDE parameter estimation under subsampling.

An Ornstein-Uhlenbeck process  dx = -theta x dt + sigma dW  is discretized by
Euler-Maruyama with step dt, giving a linear Gaussian Markov chain

    x_{t+1} = (1 - theta dt) x_t + z_t,   z_t ~ N(0, sigma^2 dt),

i.e. a chain-structured GaussianDAG with parameters *tied across time*. Data
are available only every k dt (k = 1, 2, 5, ...). Three questions:

(a) **CRB vs sampling interval.** How does the declared accuracy (Slepian-Bangs
    Cramer-Rao bound) of (theta, sigma) degrade as the observation grid is
    subsampled, with the time horizon held fixed? The drift theta and the
    diffusion sigma degrade at very different rates, and at the coarsest grids
    the Fisher metric's weakest eigenvector aligns with the direction that
    keeps the stationary variance sigma^2 / (2 theta) constant: only that
    combination remains identifiable.

(b) **Is the declaration honest?** For selected (k, N), fit (theta, sigma) by
    tied-parameter maximum likelihood (`fit_gradient_custom`, LBFGS, initialized
    at the truth) on R independent datasets of N trajectories sampled from the
    chain itself (no model mismatch), and compare the empirical standard errors
    with the CRB prediction SE_1 / sqrt(N). On the fine grids the MLE attains
    the bound; on the coarsest grid with few trajectories it does **not** yet
    (the CRB is an asymptotic bound and the AR-type small-sample bias is real)
    — increasing N restores attainment. Pre-asymptotically the theta-hat
    distribution is heavy-tailed, so both the raw-std and a robust (IQR-based)
    spread are reported, along with every raw estimate.

(c) **What the CRB does not declare.** The CRB bounds variance, not bias. Data
    from the *exactly* discretized OU transition (a = e^{-theta dt}) fitted by
    the Euler-Maruyama model leave a population-level bias O(dt) in theta and
    sigma, computed here in closed form via `local_mle_from_cov` on the exact
    covariance (no sampling noise). The exact-transition model has zero bias.

Results -> results/exp7.json.  Run: uv run python experiments/exp7_sde_crb_sampling.py
(Part (b) runs 3 x R LBFGS fits; expect a few minutes.)
"""

from __future__ import annotations

import json
import math
import os

import torch

import gaussian_bn as gbn
from gaussian_bn.training import ObsPattern, fit_gradient_custom

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp7.json")
SEED = 20260612
torch.set_default_dtype(torch.float64)

# OU truth and Euler-Maruyama grid (horizon T*dt is held fixed throughout (a,b))
THETA0, SIGMA0 = 1.0, 0.5
DT, T = 0.1, 50                      # 51 nodes, horizon 5.0

K_LIST = [1, 2, 5, 10, 25, 50]       # observe every k-th grid point
VALIDATE_CASES = [(1, 8), (5, 8), (10, 8), (10, 40)]  # (k, trajectories per dataset)
R = 120                              # replicate datasets per case
BIAS_DTS = [0.4, 0.2, 0.1, 0.05, 0.025]
BIAS_T = 40                          # transitions used in the bias study


def gen(seed):
    g = torch.Generator(); g.manual_seed(seed); return g


def em_chain(th: torch.Tensor, sg: torch.Tensor, dt: float, T: int) -> gbn.GaussianDAG:
    """Euler-Maruyama OU chain, started at the stationary variance sigma^2/(2 theta)."""
    a = (1.0 - th * dt).reshape(1, 1)
    q = (sg**2 * dt).reshape(1, 1)
    return gbn.GaussianDAG([1] * (T + 1), {(t, t + 1): a for t in range(T)},
                           [(sg**2 / (2.0 * th)).reshape(1, 1)] + [q] * T,
                           validate=False)


def exact_chain(th: float, sg: float, dt: float, T: int) -> gbn.GaussianDAG:
    """Exactly discretized OU transition: a = e^{-theta dt}, q = sigma^2 (1-a^2)/(2 theta)."""
    a = torch.tensor(math.exp(-th * dt)).reshape(1, 1)
    q = torch.tensor(sg**2 * (1.0 - math.exp(-2.0 * th * dt)) / (2.0 * th)).reshape(1, 1)
    return gbn.GaussianDAG([1] * (T + 1), {(t, t + 1): a for t in range(T)},
                           [torch.tensor(sg**2 / (2.0 * th)).reshape(1, 1)] + [q] * T,
                           validate=False)


# --------------------------------------------------------------------------
# (a) CRB of (theta, sigma) as a function of the sampling interval k
# --------------------------------------------------------------------------
def crb_vs_k():
    eta0 = torch.tensor([THETA0, SIGMA0])
    # tangent of {sigma^2 / (2 theta) = const} at the truth (normalized)
    grad_v = torch.tensor([-SIGMA0**2 / (2 * THETA0**2), SIGMA0 / THETA0])
    tang = torch.tensor([grad_v[1], -grad_v[0]]); tang = tang / tang.norm()

    rows = []
    for k in K_LIST:
        observed = list(range(0, T + 1, k))

        def K_of_eta(eta):
            return gbn.marginal(em_chain(eta[0], eta[1], DT, T), observed)

        F, w, U = gbn.fisher_metric(K_of_eta, eta0, method="fd", fd_eps=1e-6)
        se = torch.sqrt(torch.diagonal(torch.linalg.inv(F)))
        weak = U[:, 0]
        rows.append({
            "k": k, "num_obs": len(observed),
            "se_theta": float(se[0]), "se_sigma": float(se[1]),
            "cond": float(w.max() / w.min()),
            "fisher_eigvals": [float(x) for x in w],
            "weak_eigvec": [float(x) for x in weak],
            "align_const_stat_var": float(torch.abs(weak @ tang)),
        })
    return rows


# --------------------------------------------------------------------------
# (b) empirical MLE standard errors vs the CRB declaration
# --------------------------------------------------------------------------
JITTER = 1e-10                       # PD floor inside the likelihood solves


def fit_one(X: torch.Tensor, observed: list[int],
            seed_eta: torch.Tensor) -> tuple[float, float, bool]:
    """Tied-parameter MLE of (theta, sigma) on one dataset (log-parametrized).

    Fast path: LBFGS with a strong-Wolfe line search. On the coarsest grids the
    line search can overshoot into a numerically indefinite region (Cholesky
    failure); those rare datasets are refitted from the same start with plain
    Adam (bounded steps, no line search). Returns ``(theta, sigma, retried)``.
    """
    pat = [ObsPattern(tuple(observed), X[:, observed])]   # scalar nodes: index = column

    def run(optimizer: str, lr: float, num_iters: int) -> tuple[float, float]:
        eta = seed_eta.clone().requires_grad_(True)

        def build_model():
            return em_chain(torch.exp(eta[0]), torch.exp(eta[1]), DT, T)

        fit_gradient_custom([eta], build_model, pat, optimizer=optimizer,
                            lr=lr, num_iters=num_iters, jitter=JITTER)
        with torch.no_grad():
            return float(torch.exp(eta[0])), float(torch.exp(eta[1]))

    try:
        th, sg = run("lbfgs", lr=0.5, num_iters=60)
        return th, sg, False
    except torch.linalg.LinAlgError:
        th, sg = run("adam", lr=0.02, num_iters=400)
        return th, sg, True


def validate(crb_rows):
    m_true = em_chain(torch.tensor(THETA0), torch.tensor(SIGMA0), DT, T)
    eta_init = torch.log(torch.tensor([THETA0, SIGMA0]))
    se_by_k = {r["k"]: (r["se_theta"], r["se_sigma"]) for r in crb_rows}

    cases = []
    for case_idx, (k, n_paths) in enumerate(VALIDATE_CASES):
        observed = list(range(0, T + 1, k))
        est = torch.zeros(R, 2)
        retries = 0
        for r in range(R):
            X = gbn.sample(m_true, n_paths, gen(SEED + 100_000 * case_idx + r))
            th, sg, retried = fit_one(X, observed, eta_init)
            est[r] = torch.tensor([th, sg])
            retries += int(retried)
            if (r + 1) % 40 == 0:
                print(f"  [k={k}, N={n_paths}] {r + 1}/{R} fits ({retries} Adam retries)")
        emp_se = est.std(dim=0)
        # robust spread: IQR / 1.349 (= SD for a Gaussian). The theta-hat
        # distribution is heavy-tailed on coarse grids, where the raw std is
        # dominated by a few excursions; both readings are reported.
        q1, q3 = est.quantile(0.25, dim=0), est.quantile(0.75, dim=0)
        rob_se = (q3 - q1) / 1.349
        bias = est.mean(dim=0) - torch.tensor([THETA0, SIGMA0])
        crb_se = torch.tensor(se_by_k[k]) / math.sqrt(n_paths)   # N trajectories
        cases.append({
            "k": k, "num_obs": len(observed), "R": R, "N_paths": n_paths,
            "adam_retries": retries, "jitter": JITTER,
            "crb_se_theta": float(crb_se[0]), "crb_se_sigma": float(crb_se[1]),
            "emp_se_theta": float(emp_se[0]), "emp_se_sigma": float(emp_se[1]),
            "robust_se_theta": float(rob_se[0]), "robust_se_sigma": float(rob_se[1]),
            "ratio_theta": float(emp_se[0] / crb_se[0]),
            "ratio_sigma": float(emp_se[1] / crb_se[1]),
            "robust_ratio_theta": float(rob_se[0] / crb_se[0]),
            "robust_ratio_sigma": float(rob_se[1] / crb_se[1]),
            "bias_theta": float(bias[0]), "bias_sigma": float(bias[1]),
            "estimates": est.tolist(),     # raw (theta_hat, sigma_hat) per replicate
        })
    return cases


# --------------------------------------------------------------------------
# (c) Euler-Maruyama discretization bias (population level, closed form)
# --------------------------------------------------------------------------
def discretization_bias():
    rows = []
    for dt in BIAS_DTS:
        # exact-OU covariance as the population "data"
        K_exact = gbn.k_full(exact_chain(THETA0, SIGMA0, dt, BIAS_T))
        template = em_chain(torch.tensor(THETA0), torch.tensor(SIGMA0), dt, BIAS_T)
        eh, nh = gbn.local_mle_from_cov(template, K_exact)   # population MLE (full obs.)

        a_hats = torch.tensor([float(eh[(t, t + 1)]) for t in range(BIAS_T)])
        q_hats = torch.tensor([float(nh[t]) for t in range(1, BIAS_T + 1)])
        th_em = float((1.0 - a_hats.mean()) / dt)            # EM reading of the AR weight
        sg_em = float(torch.sqrt(q_hats.mean() / dt))
        th_exact = float(-torch.log(a_hats.mean()) / dt)     # exact-transition reading
        rows.append({
            "dt": dt,
            "theta_hat_em": th_em, "theta_bias_em": th_em - THETA0,
            "sigma_hat_em": sg_em, "sigma_bias_em": sg_em - SIGMA0,
            "theta_hat_exact": th_exact, "theta_bias_exact": th_exact - THETA0,
            "theta_em_analytic": (1.0 - math.exp(-THETA0 * dt)) / dt,
            "edge_spread": float(a_hats.max() - a_hats.min()),
        })
    return rows


def main():
    print(f"OU: theta={THETA0}, sigma={SIGMA0}, dt={DT}, horizon={T * DT}\n")

    print("(a) CRB per single trajectory vs sampling interval k")
    crb_rows = crb_vs_k()
    print(f"{'k':>4} {'#obs':>5} {'SE(theta)':>10} {'SE(sigma)':>10} "
          f"{'cond(F)':>10} {'align':>7}")
    for r in crb_rows:
        print(f"{r['k']:>4} {r['num_obs']:>5} {r['se_theta']:>10.4f} "
              f"{r['se_sigma']:>10.4f} {r['cond']:>10.1f} "
              f"{r['align_const_stat_var']:>7.3f}")

    print(f"\n(b) empirical MLE SE vs CRB declaration ({R} replicate datasets)")
    val_rows = validate(crb_rows)
    print(f"{'k':>4} {'N':>4} {'std/CRB th':>11} {'rob/CRB th':>11} "
          f"{'std/CRB sg':>11} {'rob/CRB sg':>11} {'bias(th)':>10}")
    for r in val_rows:
        print(f"{r['k']:>4} {r['N_paths']:>4} {r['ratio_theta']:>11.3f} "
              f"{r['robust_ratio_theta']:>11.3f} {r['ratio_sigma']:>11.3f} "
              f"{r['robust_ratio_sigma']:>11.3f} {r['bias_theta']:>10.4f}")
    print(f"(Gaussian Monte-Carlo error on the ratios ~ {1.0 / math.sqrt(2 * R):.3f}; "
          "the std ratio is tail-sensitive on coarse grids)")

    print("\n(c) Euler-Maruyama population bias vs dt (CRB does not declare this)")
    bias_rows = discretization_bias()
    print(f"{'dt':>7} {'theta_hat(EM)':>14} {'bias(EM)':>10} "
          f"{'bias(exact)':>12} {'sigma_hat(EM)':>14}")
    for r in bias_rows:
        print(f"{r['dt']:>7} {r['theta_hat_em']:>14.5f} {r['theta_bias_em']:>10.5f} "
              f"{r['theta_bias_exact']:>12.2e} {r['sigma_hat_em']:>14.5f}")

    results = {
        "seed": SEED,
        "model": {"theta0": THETA0, "sigma0": SIGMA0, "dt": DT, "T": T,
                  "horizon": T * DT},
        "crb_vs_k": crb_rows,
        "validation": val_rows,
        "discretization_bias": {"T": BIAS_T, "rows": bias_rows},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()
