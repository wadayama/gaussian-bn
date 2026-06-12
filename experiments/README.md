# experiments — applied experiments

Six self-contained experiments built on the `gaussian_bn` public API. Each
writes its numerical results to `results/<name>.json` (the **source of truth**),
and a combined LaTeX summary is generated from those files — no number is
hand-edited or synthetic.

Run each:

```bash
uv run python experiments/exp1_hidden_em.py
uv run python experiments/exp2_sensor_placement.py
uv run python experiments/exp3_interventional_ig.py
uv run python experiments/exp5_structure_learning.py
uv run python experiments/exp6_crb_reliability.py
uv run python experiments/exp7_sde_crb_sampling.py
```

Then build the summary PDF (requires a LaTeX install with `lualatex` + `luatexja`):

```bash
uv run python experiments/make_summary.py
lualatex -interaction=nonstopmode experiments_summary.tex   # run twice for plot refs
```

| Script | Theme (research note) | What it shows |
| --- | --- | --- |
| `exp1_hidden_em.py` | Hidden-node estimation (Exp 2 & 3) | Diamond with hidden middle nodes, observed only at root/sink. EM, Adam, and LBFGS all reach the same optimal NLL (`= log det S + dim`) and recover the observed covariance; the Fisher rank flags the 2 latent-scale gauge directions. |
| `exp2_sensor_placement.py` | Sensor placement / Theme C (Exp 4) | D-optimal greedy selection of observed nodes maximizing `logdet G^{(O)}`; the edge-identifiability rank reaches full at budget 3 of 6 nodes. Honestly reports that E-optimal greedy can be suboptimal (the criterion is not submodular). |
| `exp3_interventional_ig.py` | Interventional information geometry / Theme D | Confounded triangle: observational `I(V1;V2)` vs post-`do` `I^{do}(V1;V2)`. At zero direct effect the entire association is confounding (`I^{do}=0`); as the direct edge grows, `I^{do}` tracks the causal effect, and the gap isolates confounding. |
| `exp5_structure_learning.py` | Structure learning / Theme E (Exp 5) | From a fully connected supergraph, K-recursion marginal likelihood + **group-sparsity** proximal gradient zeros entire spurious edge blocks; penalized selection + unpenalized refit + BIC recovers the true DAG exactly (precision = recall = 1.0). |
| `exp6_crb_reliability.py` | Estimator reliability (Slepian–Bangs CRB) | The analytic Fisher gives the Cramér–Rao bound `(N F)^{-1}`; over 4000 MLE trials the empirical covariance matches the CRB (≈ Monte-Carlo error) and the MLE attains it. The partial-observation latent-gauge case is correctly flagged non-identifiable (`SE = ∞`). |
| `exp7_sde_crb_sampling.py` | Accuracy-declared SDE estimation under subsampling | An OU process via Euler–Maruyama is a tied-parameter Gaussian chain. The CRB declares how `(θ, σ)` accuracy degrades when observing every `k·dt` (σ degrades much faster; at the coarsest grids only the stationary-variance combination `σ²/2θ` stays identifiable — the weak Fisher eigenvector aligns with it). Tied-parameter MLE over independent replicates attains the declared SE, and the closed-form population fit exposes the `O(dt)` Euler–Maruyama bias that the CRB does *not* declare. |

Output files:

```
experiments/
├── exp{1,2,3,5,6,7}_*.py        # run scripts
├── results/exp{1,2,3,5,6,7}.json  # numerical results (committed source of truth)
├── make_summary.py           # results/{1,2,3,5}.json -> experiments_summary.tex
└── experiments_summary.pdf   # 4-experiment summary (plots from real trajectories)
```

These are research experiments (not unit tests). For the durable correctness
suite see [`../tests/`](../tests/); for tutorial-scale scripts see
[`../examples/`](../examples/).
