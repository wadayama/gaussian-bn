# tests — pytest suite

Run from the repository root:

```bash
uv run pytest                 # full suite (179 tests)
uv run pytest -m "not slow"   # fast subset (skips large-N Monte-Carlo / sampled AD-EM)
uv run pytest -k fd_vs_autograd   # gradient cross-checks only
```

Conventions (mirroring the parent `gaussian-dag` suite): pytest, no fixtures,
module-local seeded generators, closed-form recomputation as ground truth,
descriptive `test_<feature>_<context>` names. Shared helpers live in
[`_helpers.py`](_helpers.py) (seeded generators, topology builders, and
**independent** oracles: `closed_form_full_cov`, `precision_conditional`,
`empirical_regression`, `fd_jacobian`). The base seed is `GLOBAL_SEED =
20260610`, fixed so the suite is fully reproducible (e.g. the chain/collider CMI
values are deterministic across runs).

| File | Covers |
| --- | --- |
| `_helpers.py` | Seeded generators, topology builders, ground-truth oracles, tolerances. |
| `test_krecursion.py` | K-recursion vs the `(I−A)^{-1}` closed form (diamond, random, multi-root, vector, single-node); Hermitian flip; topological / root validation. |
| `test_model.py` | `GaussianDAG` offsets/indexing/validation; `k_full`; `pack`/`unpack` round-trips and PD-preserving noise. |
| `test_information.py` | `logdet_hpd`; marginal; Schur vs precision-form conditioning; MI/CMI (chain + collider d-separation); solve-vs-inv equivalence; Hermitian/PSD of returned covariances. |
| `test_sampling.py` | Sample shape/reproducibility; Monte-Carlo covariance convergence (real & complex). |
| `test_training_local.py` | Local-regression MLE exact recovery from population covariance + sampled consistency; ridge. |
| `test_training_nll.py` | `gaussian_nll` optimum / value; multi-pattern likelihood; `fit_gradient` observed-covariance recovery. |
| `test_training_em.py` | EM monotonicity; observed-covariance recovery; posterior moments; population fixed point. |
| `test_identifiability.py` | Fisher PSD/symmetry; latent-scale gauge rank & null direction; autograd-vs-fd; `identifiability_report`. |
| `test_reliability.py` | Slepian–Bangs Fisher = `edge_fisher`; CRB = `(N F)^{-1}` and `∝ 1/N`; gauge → `SE = ∞`; empirical MLE covariance attains the CRB. |
| `test_intervention.py` | `do_hard`/`do_soft` = manual edge-deletion + recompute; intervention breaks dependence; validation. |
| `test_design.py` | D-/E-optimal score values; greedy monotonicity; greedy vs exhaustive; validation. |
| `test_optimize.py` | PGA ascent + projector; natural-gradient step (identity-metric reduction, one-step-exact, monotonicity). |
| `test_projections.py` | Frobenius-ball / total-power projection correctness and idempotence. |
| `test_gradients.py` | Finite-difference vs autograd for `K_OO` and the marginal-NLL gradient (edge & noise params). |
| `test_invariants.py` | Property tests over random parameters: marginalization consistency, MI/CMI symmetry, CMI = 0 under d-separation, conditioning reduces covariance (PSD order), data-processing, MI chain rule. |

Exact-formula tests use `atol = 1e-10`; identities `1e-12`; invariants `1e-9`;
FD-vs-autograd `rel_err < 1e-6`; Monte-Carlo covariance `< 0.02` Frobenius
(`< 0.03` for complex). Dtype-generic tests run under both `float64` and
`complex128`.
