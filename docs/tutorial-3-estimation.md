# Tutorial 3 — Estimation: local regression, marginal likelihood, and EM

We estimate the edge matrices and noise covariances from data. The right tool
depends on whether all nodes are observed.

## Generate data

```python
import torch
import gaussian_bn as gbn

m = gbn.GaussianDAG(
    dims=[1, 1, 1, 1],
    edges={(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]},
    noise=[[[1.0]], [[0.4]], [[0.5]], [[0.3]]],
)
X = gbn.sample(m, 200_000, torch.Generator().manual_seed(1))   # (N, D) samples
```

## Full observation → closed-form MLE

When every node is observed, the likelihood factorizes and the MLE is node-wise
Gaussian regression — no iteration:

```python
edges_hat, noise_hat = gbn.fit_local_regression(m, X)          # closed form
# edges_hat[(i, j)] ≈ true A_ji ;  noise_hat[j] ≈ true Σ_j
```

`ridge=λ` adds a `(K_UU + λ I)^{-1}` shrinkage for ill-conditioned parents.

## Partial observation → marginal likelihood

With hidden nodes, fit the **observed-block** Gaussian NLL
`L(η) = log det K_{OO}(η) + tr(K_{OO}(η)^{-1} S)`, which is differentiable
through the K-recursion. Two routes:

### (a) Gradient training

```python
observed = [0, 3]                                  # hide the middle nodes 1, 2
oi = m.node_index(observed)
S = (X[:, oi].mH @ X[:, oi]) / X.shape[0]
patterns = [gbn.ObsPattern(tuple(observed), X[:, oi])]

# start from a random model with the same structure
g = torch.Generator().manual_seed(2)
init = gbn.GaussianDAG(m.dims,
    {k: torch.randn(1, 1, generator=g, dtype=torch.float64) for k in m.edges},
    [torch.tensor([[0.7]], dtype=torch.float64) for _ in range(4)])

fitted, nll = gbn.fit_gradient(init, patterns, optimizer="lbfgs", num_iters=500)
```

`fit_gradient` packs the model into unconstrained autograd leaves (with a
positive-definite-preserving noise parametrization), runs Adam or LBFGS, and
returns the fitted model plus the NLL history. An optional
`regularizer=lambda model: …` adds penalties (e.g. group sparsity).

**Shared / factored edges.** When the edges are *not* free and independent — a
relay factor shared across a node's outgoing edges (`A_{ji} = H_{ji} F_i`), a
low-rank edge (`A = U V^H`), or tied blocks — use `fit_gradient_custom`: you pass
the leaf parameters and a closure that builds the model from them, and autograd
back-propagates the marginal likelihood through any edge construction. See
`examples/03_learn_shared_factor.py`.

### (b) EM

```python
fitted_em, nll_history = gbn.em_fit(m, X, observed, num_iters=60)
```

EM uses the K-recursion to form the E-step posterior moments
(`posterior_full_moments`) and a node-wise regression M-step. Its NLL decreases
monotonically.

## What is recovered?

With hidden nodes the **observed covariance** `K_{OO}` is recovered (both routes
reach the same optimal NLL `= log det S + dim`), but the edge matrices
themselves may carry a latent gauge — exactly what the next tutorial diagnoses.

Next: [identifiability, intervention, and design](tutorial-4-identifiability-and-design.md).
