# MATH.md — implementation-side mathematics

This note states the mathematics behind `gaussian-bn`, written against the
library's 0-based indexing and API function names. It is an implementation
companion, not a formal paper. Throughout, `(·)^H` is the conjugate (Hermitian)
transpose (`.mH`), which reduces to an ordinary transpose for real models.

## 1. Model

A linear Gaussian Bayesian network on nodes `0, …, M−1` in topological order:

```
V_j = Σ_{i ∈ Pa(j)} A_{ji} V_i + Z_j,     Z_j ~ N(0, Σ_j),    (1)
```

with `V_j ∈ ℂ^{d_j}` (or `ℝ^{d_j}`), edge matrices `A_{ji} ∈ ℂ^{d_j × d_i}`, and
mutually independent innovations `Z_j`. Parentless nodes are **roots**; for a
root `r`, equation (1) reduces to `V_r = Z_r ~ N(0, Σ_r)`. The stacked vector
`V_all = (V_0^T, …, V_{M−1}^T)^T` is zero-mean Gaussian with covariance `K_all`.

`GaussianDAG(dims, edges, noise)` stores this model; `edges[(i,j)] = A_{ji}` and
`noise[j] = Σ_j` (root covariance for roots).

**Means decouple.** An optional per-node offset `c_j` makes the model affine,
`V_j = c_j + sum_i A_{ji} V_i + Z_j`. The mean and covariance then separate
completely: the covariance `K_all` is unchanged (the K-recursion below is
untouched), while the marginal means follow a parallel, cheaper **mean
recursion** `m_j = c_j + sum_{i in Pa(j)} A_{ji} m_i`, i.e. `m_all = (I-A)^{-1} c`
(`mean_all`). Mutual information, conditional MI, conditional independence, and
the covariance Fisher metric depend only on `K` and are mean-invariant. A
zero-mean model (`mean=None`) is the default and a numeric no-op.

## 2. The K-recursion

The node-pair covariance blocks are `K_{jk} = E[V_j V_k^H]`. Substituting (1)
and using independence of the innovations gives, in topological order,

```
K_{jk} = Σ_{i ∈ Pa(j)} A_{ji} K_{ik}                              (j > k)      (2a)
K_{jj} = Σ_{i, i' ∈ Pa(j)} A_{ji} K_{ii'} A_{ji'}^H + Σ_j                       (2b)
```

with `K_{rr} = Σ_r` for roots. Only matrix products, sums, and Hermitian
transposes appear, so the map `η = {A_{ji}, Σ_j} ↦ K_all(η)` is a smooth,
autograd-differentiable computation graph (`compute_k_blocks`).

**Canonical storage.** Only `K_{jk}` for `j ≥ k` is stored; the rest follow from
`K_{ab} = K_{ba}^H`, applied on read by `get_K`. The crucial point of (2b) is the
**parent cross-covariance** `K_{ii'}`: at a merging node, propagating only
self-covariances would be wrong.

**Closed-form cross-check.** Writing (1) as `V_all = (I − 𝒜)^{-1} Z` with `𝒜` the
strictly-block-lower-triangular edge matrix gives
`K_all = (I − 𝒜)^{-1} Σ (I − 𝒜)^{-H}` (`full_covariance_closed_form`, a
non-differentiable test oracle).

## 3. Inference

All queries read sub-blocks of `K_all` (assembled by `k_full`).

**Marginal.** `V_A ~ N(0, K_AA)` (`marginal`).

**Conditioning (Schur complement).** For node sets `A, B` with `K_BB ≻ 0`,

```
E[V_A | V_B = b] = m_A + K_AB K_BB^{-1} (b − m_B)         (conditional_mean)
Cov(V_A | V_B)   = K_AA − K_AB K_BB^{-1} K_BA             (conditional_covariance)
```

with `m = mean_all(model)` (`m_A = m_B = 0` for a zero-mean model, recovering
`K_AB K_BB^{-1} b`). The library evaluates `K_BB^{-1}(·)` with a linear solve
(`solve_psd`) and re-symmetrizes the conditional covariance (`schur_complement`).

**Mutual information / CMI (log-det).** With the real-Gaussian `½` convention
(nats),

```
I(V_A; V_B)      = ½ ( log det K_AA  − log det K_{A|B} )                (mutual_information)
I(V_A; V_B | V_C) = ½ ( log det K_{A|C} − log det K_{A|BC} )  (conditional_mutual_information)
```

where `K_{A|C} = K_AA − K_AC K_CC^{-1} K_CA`, etc. Log-determinants of Hermitian
PD matrices use a Cholesky factorization (`logdet_hpd`), never a general
`slogdet`. `I(V_A;V_B|V_C) = 0` certifies conditional independence and is
consistent with graphical d-separation.

## 4. Estimation

### 4.1 Full observation — node-wise MLE

When all nodes are observed the joint NLL factorizes per node, and the MLE is
ordinary multivariate Gaussian regression of `V_j` on its parents `U_j`:

```
B̂_j = K̂_{jU} K̂_{UU}^{-1},     Σ̂_j = K̂_{jj} − B̂_j K̂_{Uj}                    (3)
```

with sample (or expected) covariance blocks `K̂`. `fit_local_regression`
(equivalently `local_mle_from_cov` on a covariance matrix) implements (3) with a
solve and an optional ridge `(K̂_{UU} + λ I)^{-1}`.

### 4.2 Partial observation — marginal likelihood

With hidden nodes, observe `O`. The observed-block negative log-likelihood
(up to a constant) is

```
L(η) = log det K_{OO}(η) + tr( K_{OO}(η)^{-1} S ),     S = (1/N) Σ_n y_O^{(n)} y_O^{(n)H},   (4)
```

(`gaussian_nll`; `marginal_likelihood` generalizes to per-sample observation
patterns / missing data). `L` is differentiable through the K-recursion, so
`fit_gradient` maximizes it by Adam or LBFGS over a positive-definite-preserving
parametrization (`pack`/`unpack`). The minimizer of (4) over a flexible model is
`K_{OO} = S`, with `L = log det S + dim`.

Because the K-recursion differentiates through *any* construction of the edge
matrices, the edges need not be free and independent. Building each `A_{ji}` from
shared or structured factors — a relay factor shared across a node's outgoing
edges (`A_{ji} = H_{ji} F_i`), a low-rank edge (`A = U V^H`), or tied blocks —
and minimizing (4) over those factors is `fit_gradient_custom`; autograd supplies
the chain rule. A regularizer `R(η)` added to (4) covers ridge, group-sparsity
(structure pruning), low-rank, and parameter-sharing penalties.

### 4.3 EM

`em_fit` alternates:

- **E-step.** With current `η`, the posterior moments under `V_O = y_O` are
  `m_n = K_{·,O} K_{OO}^{-1} y_O^{(n)}` and
  `C = K_{··} − K_{·,O} K_{OO}^{-1} K_{O,·}`, giving the expected sufficient
  statistic `K̃ = C + (1/N) Σ_n m_n m_n^H` (`posterior_full_moments`).
- **M-step.** Apply the node-wise regression (3) to `K̃`.

The observed-data NLL decreases monotonically.

## 5. Identifiability — the pullback Fisher metric

For a zero-mean Gaussian, the Fisher information of the covariance induces, on
the parameter space through `η ↦ K_{OO}(η)`, the pullback metric

```
G^{(O)}_{ab}(η) = ½ tr[ K_{OO}^{-1} (∂K_{OO}/∂η_a) K_{OO}^{-1} (∂K_{OO}/∂η_b) ].   (5)
```

`fisher_metric` evaluates (5) with an autograd Jacobian of `K_{OO}` (or finite
differences); `edge_fisher` specializes `η` to chosen edge entries. The
parameter `η` is **locally identifiable** from `O` iff `rank G^{(O)} = q`. A rank
deficiency exposes a gauge: e.g. a hidden node `H` feeding observed `Y_1 = aH`,
`Y_2 = bH` admits the scale gauge `(a, b, Var H) → (a/t, b/t, t² Var H)`, leaving
`K_{OO}` invariant — `identifiability_report` returns the rank, condition number,
and the null directions in edge space.

## 5b. Reliability — Slepian–Bangs Fisher and the Cramér–Rao bound

For a zero-mean Gaussian observation `V_O ~ N(0, K_OO(θ))`, the **Slepian–Bangs**
formula gives the per-observation Fisher information in closed form, and for the
zero-mean case it is exactly the pullback metric (5):

```
F_{ab}(θ) = ½ tr[ K_OO^{-1} (∂K_OO/∂θ_a) K_OO^{-1} (∂K_OO/∂θ_b) ].   (Slepian–Bangs, zero-mean)
```

For an affine model the full Slepian–Bangs information adds the mean term
`(∂m_O/∂θ_a)^H K_OO^{-1} (∂m_O/∂θ_b)` (`include_mean=True`); it is appropriate
only when the offsets are *known*, since by the mean/covariance orthogonality of
Gaussian MLE an estimator that also fits the offsets has edge variance governed
by the covariance term alone (hence `include_mean=False` is the default). Because
`K_OO(θ)` comes from the K-recursion, `F` is computed
**analytically** — no sampling — which is what `parameter_fisher` /
`fisher_metric` do. For `N` i.i.d. observations the total information is `N F`,
and the **Cramér–Rao bound** is

```
Cov(θ̂) ⪰ (N F)^{-1}                                                  (CRB)
```

for any unbiased `θ̂`; the MLE/EM estimators attain it asymptotically, so
`sqrt(diag((N F)^{-1}))` are asymptotic standard errors (`crb`, `crb_report`).
When `F` is rank-deficient — a latent gauge under partial observation — the
bound is unbounded along `null(F)`: a parameter whose unit vector has a component
in `null(F)` is non-identifiable and reported with `SE = ∞`. Thus the same Fisher
object yields both the **point-estimate reliability** (finite directions) and the
**identifiability verdict** (null directions).

## 6. Intervention

A hard intervention `do(V_j)` removes the incoming edges of `j` and makes it an
independent root with a chosen covariance; a **point intervention** `do(V_j = u)`
additionally fixes the node deterministically (offset `u`, zero covariance); a
soft intervention replaces `j`'s edges and/or noise. Either way the modified
model is a new `GaussianDAG` on which the same K-recursion runs (`do_hard`,
`do_soft`). Comparing observational `I` to post-intervention `I^{do}` separates
causal from confounding association; the change in `log det G^{(O)}` measures the
intervention's effect on identifiability.

**Counterfactuals** (`counterfactual`) use Pearl's abduction–action–prediction
for the linear Gaussian SEM: (1) *abduction* infers the exogenous noise
consistent with the evidence from the posterior mean
`v = E[V_all | V_E = e]` and `z_k = v_k − c_k − sum_i A_{ki} v_i`; (2) *action*
fixes the intervened nodes; (3) *prediction* re-runs the structural equations in
topological order with the **same** abducted noise `z`. This answers "what would
`V_Q` have been, had `V_j` been `u`, given that `V_E = e` was observed".

## 7. Observation design

From (5), choose observed nodes under a budget to maximize an optimality
criterion of the edge information `G^{(O)}`:

```
D-optimal:  max_{|O| ≤ m}  log det( G^{(O)} + ε I )
E-optimal:  max_{|O| ≤ m}  λ_min( G^{(O)} + ε I )
```

`sensor_placement` offers greedy and exhaustive search. D-optimality is (near)
submodular, so greedy is near-optimal; E-optimality is not, so greedy may be
suboptimal — use exhaustive search on small instances.

## 8. Relationship to prior work

We do **not** claim the K-recursion as a new covariance algorithm or a new
statistical object. The map from local conditional parameters to the joint
covariance of a linear Gaussian network is classical, and so are its
specializations and equivalent forms:

- **Local-to-global covariance map.** Computing a Gaussian network's covariance
  from its local node parameters is classical: Gaussian influence diagrams
  translate between the two via arc reversal (Shachter & Kenley, *Gaussian
  Influence Diagrams*, Management Science 35(5), 1989), and the covariance can be
  built recursively through the precision matrix (Geiger & Heckerman, *Learning
  Gaussian Networks*, UAI 1994). The closed form `K = (I − A)^{-1} Σ (I − A)^{-H}`
  — of which the K-recursion is a topological forward-substitution evaluation —
  is standard in structural-equation modelling (Bollen, *Structural Equations
  with Latent Variables*, 1989).
- **Covariance as treks (path analysis).** Each covariance entry is a sum over
  treks (pairs of directed paths sharing a source) of edge-coefficient products
  times source variances — Wright's path analysis (*Correlation and Causation*,
  J. Agric. Research 20, 1921), formalized as the *trek rule* / *trek separation*
  by Sullivant, Talaska & Draisma (*Annals of Statistics* 38(3), 2010). The
  parent cross-covariance `K_{ii'}` (`i ≠ i'`) in the self-block (2b) is exactly
  a trek contribution through shared ancestors.
- **Chain / state-space special case.** For a Markov chain every node has a
  single parent, so the self-block recursion (2b) collapses to `A K A^H + Σ` with
  **no cross term** — the Kalman covariance-prediction recursion (Kalman 1960),
  the Rauch–Tung–Striebel smoother (1965), and the discrete Lyapunov equation.
  The general DAG goes qualitatively beyond this chain subset by carrying the
  parent cross-covariances `K_{ii'}` at merging nodes and the off-diagonal blocks
  propagated through shared ancestors (which are the trek contributions above).

**What this library contributes** is a *formulation* and a *framework*, not new
mathematics: the covariance map is realized as a single **differentiable**,
**vector/matrix-valued**, **all-pairs** forward operator (inverse-free,
topological), and the entire stack of this document — conditioning, MI/CMI,
marginal-likelihood / EM training, the Slepian–Bangs Fisher information and
Cramér–Rao bounds, interventions and counterfactuals, and observation design — is
built on that one operator and composes with automatic differentiation. The
recursion is standard linear algebra; the value is the unified differentiable
backend it provides.

---

For the conventions (indexing, dtype, units, numerical hardening) see
[`README.md`](README.md#conventions); for the function reference see the
[Public API](README.md#public-api).
