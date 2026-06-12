# Tutorials

A four-part walkthrough of `gaussian-bn`, from installation to identifiability
and experiment design. Each part is self-contained with runnable code; run any
snippet with `uv run python - <<'PY' … PY` or paste it into a script.

1. [Installation and your first inference](tutorial-1-installation-and-inference.md)
   — install with `uv`, build a two-node BN, compute mutual information.
2. [Building a Gaussian BN and reading K-blocks](tutorial-2-building-a-bn.md)
   — topological order, multiple roots, vector nodes, `k_full` and `get_K`,
   marginalization and conditioning.
3. [Estimation: local regression, marginal likelihood, and EM](tutorial-3-estimation.md)
   — full-observation MLE, the observed-data NLL, gradient training, and EM for
   hidden nodes.
4. [Identifiability, intervention, and sensor placement](tutorial-4-identifiability-and-design.md)
   — the pullback Fisher metric, latent gauges, do-operations, and D-/E-optimal
   observation design.

For the underlying mathematics see [`../MATH.md`](../MATH.md); for the full API
see [`../README.md`](../README.md#public-api).
