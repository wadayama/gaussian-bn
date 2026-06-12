# examples — quick-start scripts

Short, self-contained scripts illustrating the core API. Each runs in a second
on CPU and prints its results (no files written). For larger applied
experiments with saved results and a LaTeX summary, see
[`../experiments/`](../experiments/); for a guided walkthrough, see
[`../docs/`](../docs/).

```bash
uv run python examples/01_inference_cmi.py
uv run python examples/02_estimation_em.py
uv run python examples/03_learn_shared_factor.py
uv run python examples/04_counterfactual.py
```

| Script | Demonstrates |
| --- | --- |
| `01_inference_cmi.py` | Build a diamond Gaussian BN; marginal covariance; mutual information; conditional mutual information showing both a d-separation (`I(V1;V2\|V0)=0`) and explaining-away at a collider (`I(V1;V2\|V3)>0`); Gaussian posterior mean/variance. |
| `02_estimation_em.py` | Full-observation closed-form MLE (recovers true edges); hidden-node EM (recovers the observed covariance); Fisher-based identifiability report revealing the latent edge gauge. |
| `03_learn_shared_factor.py` | Learn a single relay factor `F` shared across two edges (`A_{0->1}=H1 F`, `A_{0->2}=H2 F`) from data with `fit_gradient_custom` — parameter sharing through a differentiable closure. |
| `04_counterfactual.py` | An affine causal chain with node offsets: marginal means, a point intervention `do(M=10)`, and a counterfactual ("had X been different, what would Y have been?") via abduction–action–prediction. |
