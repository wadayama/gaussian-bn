# Tutorial 1 — Installation and your first inference

## Install

`gaussian-bn` is managed with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/wadayama/gaussian-bn.git
cd gaussian-bn
uv sync
uv run pytest -m "not slow"     # confirm the install (fast subset)
```

`uv sync` builds `.venv/` from the locked `uv.lock`. Run anything inside the
environment with `uv run python …`.

## The model

A linear Gaussian Bayesian network is a DAG whose nodes are vector-valued
Gaussians and whose edges are linear maps:

```
V_j = Σ_{i ∈ Pa(j)} A_{ji} V_i + Z_j,   Z_j ~ N(0, Σ_j)
```

Parentless nodes are **roots** (their `Σ_j` is the root covariance). Everything
the library computes flows from the node-pair covariances `K_{jk} = E[V_j V_k^H]`,
which the *K-recursion* builds in one topological sweep.

## A two-node channel: `Y = A X + Z`

The simplest BN: a root `X = V_0 ~ N(0, Σ_X)` and a child `Y = V_1 = A X + Z`.

```python
import torch
import gaussian_bn as gbn

A = torch.tensor([[1.5, 0.2], [0.0, 0.8]], dtype=torch.float64)   # edge 0 -> 1
Sigma_X = torch.eye(2, dtype=torch.float64)                       # root covariance
Sigma_Z = 0.3 * torch.eye(2, dtype=torch.float64)                 # innovation noise

m = gbn.GaussianDAG(dims=[2, 2], edges={(0, 1): A}, noise=[Sigma_X, Sigma_Z])

Kf = gbn.k_full(m)          # 4x4 joint covariance of (X, Y)
mi = gbn.mutual_information(m, [0], [1], Kf)
print(f"I(X; Y) = {mi.item():.4f} nats")
```

`mutual_information` uses the log-determinant form
`I(X;Y) = ½(log det Σ_Y − log det Σ_{Y|X})` (nats), computed from the K-blocks
via a Cholesky log-det and a Schur complement — no explicit matrix inverse.

## Units and conventions

- Mutual information is in **nats** (natural log): the `½ log-det` convention
  for real models, `1 log-det` for circular-complex models.
- Node indices are 0-based and topological: every edge `(i, j)` has `i < j`.
- The default dtype is `float64`; pass `complex128` tensors for circular-complex
  models and the same code runs unchanged.

Next: [building larger DAGs and reading K-blocks](tutorial-2-building-a-bn.md).
