# Tutorial 4 — Identifiability, intervention, and sensor placement

## Edge identifiability and the Fisher metric

Can the edge parameters be recovered from a given observation set? The pullback
Fisher metric on the observed block answers this:

```
G^{(O)}_{ab} = ½ tr[ K_OO^{-1} (∂K_OO/∂η_a) K_OO^{-1} (∂K_OO/∂η_b) ]
```

A rank deficiency is a non-identifiable direction (a latent gauge).

```python
import gaussian_bn as gbn

m = gbn.GaussianDAG(
    dims=[1, 1, 1, 1],
    edges={(0, 1): [[1.3]], (0, 2): [[-0.8]], (1, 3): [[0.9]], (2, 3): [[1.1]]},
    noise=[[[1.0]], [[0.4]], [[0.5]], [[0.3]]],
)

report = gbn.identifiability_report(
    m, edge_params=[(0, 1), (0, 2), (1, 3), (2, 3)], observed=[0, 3],
)
print(report.identifiable, report.rank, "/", report.q)   # False 2 / 4
print(report.null_directions)    # the 2 gauge directions, mapped to edge space
```

`report.identifiable` is `False`: observing only `{0, 3}` leaves two scale-gauge
directions (one per hidden node) unobservable. Observing more nodes raises the
rank — which motivates *sensor placement* below.

## Reliability: the Cramér–Rao bound

The same Fisher information is the **Slepian–Bangs** information of the observed
Gaussian, computed analytically from the K-recursion. Inverting `N · F` gives the
Cramér–Rao bound — standard errors and confidence intervals an MLE/EM estimate
attains asymptotically, with `∞` for the non-identifiable directions:

```python
# reliability of the two chain edges estimated from N = 1000 fully observed samples
chain = gbn.GaussianDAG([1, 1, 1], {(0, 1): [[0.9]], (1, 2): [[-0.7]]},
                        [[[1.0]], [[0.5]], [[0.4]]])
print(gbn.crb_report(chain, observed=[0, 1, 2], N=1000).summary())
```

`crb_report` prints each parameter's estimate, standard error, and confidence
interval. For a non-identifiable parameter set (e.g. the latent model above with
the noise also free, `noise_param="logdiag"`) the report marks the affected
parameters `SE = ∞` — the bound itself tells you what is and is not estimable.
See `experiments/exp6_crb_reliability.py` for a Monte-Carlo check that the MLE
attains the bound.

## Intervention (do-operations)

Interventions modify the DAG and re-run the K-recursion. Each returns a **new**
`GaussianDAG`; the original is untouched.

```python
# Hard intervention: cut node 1 from its parents (independent-root mechanism)
m_do = gbn.do_hard(m, node=1)

# Soft intervention: replace node 3's incoming edge from 1
import torch
m_soft = gbn.do_soft(m, node=3, edges={1: torch.zeros(1, 1, dtype=torch.float64)})

# Causal vs observational association
print(float(gbn.mutual_information(m, [1], [3])))      # observational
print(float(gbn.mutual_information(m_do, [1], [3])))   # after do(1)
```

This is the basis of *interventional information geometry*: comparing
`I` and `I^{do}` separates causal from confounding association
(see `experiments/exp3_interventional_ig.py`).

### Affine models, point interventions, and counterfactuals

Give the model per-node offsets to make it affine
(`V_j = c_j + Σ A_{ji} V_i + Z_j`); the covariance machinery is unchanged and a
parallel mean recursion carries the offsets (`mean_all`). This unlocks
point-value interventions and counterfactuals:

```python
import torch
ma = gbn.GaussianDAG([1, 1, 1], {(0, 1): [[0.8]], (1, 2): [[0.9]]},
                     [[[1.0]], [[0.5]], [[0.4]]],
                     mean=[torch.tensor([2.0]), torch.tensor([-1.0]), torch.tensor([0.5])])

m_point = gbn.do_hard(ma, node=1, value=torch.tensor([10.0]))   # do(V1 = 10)

# counterfactual: "had V0 been 5 instead of the observed 3, what would V2 be?"
cf = gbn.counterfactual(ma, evidence=[0, 1, 2],
                        evidence_values=torch.tensor([3.0, 1.5, 0.7]),
                        do={0: torch.tensor([5.0])}, query=[2])
```

The offsets can be learned from data — closed-form with
`fit_local_regression(model, X, fit_mean=True)`, or by gradient with
`fit_gradient_custom` (build the model with `mean=` leaf parameters). With known
offsets, `crb_report(..., include_mean=True)` adds the Slepian–Bangs mean term to
the reliability bound.

## Sensor placement (observation design)

Given a budget of observable nodes, which should we observe to best identify the
edges? Maximize a Fisher optimality criterion of `G^{(O)}`:

```python
res = gbn.sensor_placement(
    m, edge_params=[(0, 1), (0, 2), (1, 3), (2, 3)],
    candidate_nodes=[0, 1, 2, 3], budget=2,
    criterion="d",        # D-optimal: maximize logdet(G + eps I); "e" = E-optimal
    method="greedy",      # or "exhaustive"
)
print(res.chosen, res.score, res.trace)
```

D-optimal greedy is near-optimal here (it matches exhaustive); E-optimality is
not submodular, so greedy can be suboptimal — use `method="exhaustive"` on small
instances. See `experiments/exp2_sensor_placement.py` for a full sweep showing
the identifiability rank rising as the budget grows.

## Where to go next

- `experiments/` — applied versions of all four themes with saved results and a
  LaTeX summary.
- `MATH.md` — the K-recursion, conditioning, Fisher metric, and intervention
  written out against the API names.
