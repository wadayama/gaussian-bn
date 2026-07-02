# Tutorial 2 — Building a Gaussian BN and reading K-blocks

## Constructing a `GaussianDAG`

A model needs three things:

- `dims` — node dimensions, length `M`, in topological order;
- `edges` — `{(i, j): A_ji}` for each edge `i → j` with `i < j` and
  `A_ji` of shape `(dims[j], dims[i])`;
- `noise` — `noise[j] = Σ_j` (the root covariance for parentless nodes).

```python
import gaussian_bn as gbn

# Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (scalar nodes)
m = gbn.GaussianDAG(
    dims=[1, 1, 1, 1],
    edges={(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]},
    noise=[[[1.0]], [[0.4]], [[0.5]], [[0.3]]],
)
print(m.parents)   # {0: [], 1: [0], 2: [0], 3: [1, 2]}
print(m.roots)     # [0]
```

The constructor validates topological order, edge shapes, and that every noise
covariance is square and positive-definite; violations raise a clear
`ValueError`. dtype/device are inferred from the tensors you pass (and default
to `float64` for Python-list input).

**Multiple roots and vector nodes** are fully supported — any parentless node is
an independent root, and nodes can have different dimensions:

```python
m2 = gbn.GaussianDAG(
    dims=[2, 3, 1],                         # vector nodes of mixed size
    edges={(0, 2): [[0.5, -0.2]], (1, 2): [[0.3, 0.1, -0.4]]},   # two roots 0, 1
    noise=[ [[1,0],[0,1]], [[1,0,0],[0,1,0],[0,0,1]], [[0.2]] ],
)
print(m2.roots)    # [0, 1]
```

## Reading covariance blocks

`k_full(m)` assembles the full `D × D` covariance. For block-level access the
K-recursion stores only canonical lower-triangular blocks `K[(j, k)]` (`j ≥ k`);
`get_K` returns any block, applying the Hermitian flip `K_{ab} = K_{ba}^H`:

```python
K = m.k_blocks()
gbn.get_K(K, 3, 0)        # cross-covariance E[V3 V0^H]
gbn.get_K(K, 0, 3)        # == get_K(K, 3, 0).mH
```

The key fact that makes the diamond non-trivial: the merging node's self-block
`K_{33}` depends on the **parent cross-covariance** `K_{12}`. Propagating only
self-covariances would be wrong; the K-recursion keeps all cross-blocks.

## Marginalization and conditioning

```python
Kf = gbn.k_full(m)
gbn.marginal(m, [0, 3], Kf)                 # K over the sub-vector (V0, V3)
gbn.conditional_covariance(m, [3], [0, 1], Kf)   # Cov(V3 | V0, V1)
gbn.conditional_mean(m, [3], [0], b=[1.0], Kf=Kf)  # E[V3 | V0 = 1]
```

Conditioning is the Schur complement
`K_{A|B} = K_AA − K_AB K_BB^{-1} K_BA`, evaluated with a Cholesky factorization
and triangular solves, and re-symmetrized.

Next: [estimation from data](tutorial-3-estimation.md).
