# gaussian_bn — library source

Core library for inference, estimation, identifiability, intervention, and
sensor design on linear Gaussian Bayesian networks, all built on the
K-recursion. Every public symbol is re-exported from the top-level package and
documented (with signatures) in [`../README.md`](../README.md#public-api).

| Module | Purpose |
| --- | --- |
| `__init__.py` | Public-API re-exports (49 symbols, alphabetized in `__all__`). |
| `linalg.py` | Numerical core: `hermitianize`, `logdet_hpd`, `solve_psd` (solve, never inverse), `schur_complement` (re-symmetrized), `cholesky_psd`. Single chokepoint for the jitter / PD policy. |
| `krecursion.py` | Forward K-recursion: `compute_k_blocks` (multi-root) and the Hermitian-flip accessor `get_K`. |
| `model.py` | `GaussianDAG` container (validation, dtype/device inference, optional affine `mean` offsets, `k_blocks`), `k_full`, `mean_all`, `full_covariance_closed_form` (test oracle), and the autograd parametrization `pack` / `unpack` / `ParamPack`. |
| `inference.py` | `marginal`, `conditional_covariance`, `conditional_mean`, `mutual_information`, `conditional_mutual_information`, `sample`. |
| `training.py` | Estimation: `fit_local_regression` / `local_mle_from_cov` (+ridge), `gaussian_nll`, `marginal_likelihood` (multi-pattern), `posterior_full_moments`, `em_fit`, `fit_gradient` (free edges), `fit_gradient_custom` (shared / factored / constrained edges via a closure), `ObsPattern`. |
| `identifiability.py` | Pullback Fisher metric `fisher_metric` / `edge_fisher` and `identifiability_report` (`IdentifiabilityReport`). |
| `reliability.py` | Slepian–Bangs Fisher information and the Cramér–Rao bound: `parameter_fisher`, `crb`, `crb_report` (`CRBReport`) — standard errors, confidence intervals, and `SE = ∞` for non-identifiable directions. |
| `intervention.py` | Do-operations `do_hard` (mechanism or point `do(V_j=u)`), `do_soft`, `do_covariance`, `do_cmi` (each returns a fresh `GaussianDAG`), and `counterfactual` (abduction–action–prediction). |
| `design.py` | D-/E-optimal `sensor_placement` (greedy + exhaustive) and the score functions `d_optimal_score`, `e_optimal_score`. |
| `optimize.py` | `pga_ascent` and Fisher-preconditioned `natural_gradient_step` / `natural_gradient_ascent`. |
| `projections.py` | `project_frobenius_ball`, `project_total_power` (constrained-optimization projectors). |

## Design notes

- **Dtype/device agnostic.** No module hard-codes a dtype or device; tensors
  inherit them from their inputs. `GaussianDAG` infers dtype/device from the
  tensors it is given (falling back to `float64` for pure-Python-list input).
  The same code runs under `float64` (default) and `complex128`.
- **Numerical hardening.** Conditioning routes through `solve_psd` (a linear
  solve), never an explicit `inv`; Schur complements and self-blocks are
  re-symmetrized; `jitter` floors the PD cone where needed. The only `inv` is in
  `full_covariance_closed_form`, kept as an independent non-differentiable test
  oracle.
- **Canonical storage.** The K-recursion stores only `K[(j, k)]` for `j ≥ k`;
  reads of the upper half go through `get_K`, which applies `K_{ab} = K_{ba}^H`.
- **Multiple roots.** Any parentless node is an independent root, generalizing
  the single-root assumption of the parent `gaussian-dag` library.
- **One parametrization.** `pack` / `unpack` give a single PD-preserving,
  unconstrained autograd parametrization shared by gradient training, the Fisher
  metric, and natural-gradient optimization.

See [`../MATH.md`](../MATH.md) for the mathematics behind these functions.
