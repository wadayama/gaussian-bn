# experiments — applied experiments

Ten self-contained experiments built on the `gaussian_bn` public API. Each
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
uv run python experiments/exp8_lds_state_space.py
uv run python experiments/exp9_sensor_calibration.py
uv run python experiments/exp10_lds_skip.py
uv run python experiments/exp11_design_gradient.py
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
| `exp8_lds_state_space.py` | LDS running example (paper Sec. VI) | A vector linear Gaussian state-space model `x_n = A x_{n-1} + u_n`, `y_n = x_n + w_n` with the tied factorization `A = S T` exercised through every primitive. Marginal / smoother covariances match the Kalman prediction and RTS recursions to machine precision; the Markov property shows up as a vanishing CMI; forward sampling converges as `1/sqrt(M)`; the tied factor `T` is recovered from noisy observations of the hidden states; and the analytic Cramér–Rao bound over `T` matches the empirical MLE scatter across sample sizes `M`. Writes `lds_*.dat` (pgfplots data for the paper figure). |
| `exp9_sensor_calibration.py` | Sensor-network self-calibration (paper Sec. VI, non-chain) | A fusion DAG: two hidden latent roots feed 8 scalar sensors (every sensor a merging node). Three sensors are uncalibrated; the Fisher rank deficit over their loadings follows the theoretical gauge staircase `{3, 1, 0}` as `m = 0, 1, ≥2` calibrated references are co-observed (the O(3) stabilizer dimension), verified exhaustively over all 2^5 reference subsets. Past the threshold the CRB spreads strongly across equal-size placements, and Monte-Carlo MLE scatter matches the analytic CRB for both the best and the worst placement. Writes `calib_*.dat` (pgfplots data for the paper figure). |
| `exp10_lds_skip.py` | Skip-connected LDS (paper Sec. VI-F, non-chain) | The exp8 state-space model plus a tied skip connection `x_n = A x_{n-1} + C x_{n-2} + u_n`, so every state (n >= 3) merges two correlated parents and no first-order chain recursion applies to the graph as given. The chart's marginals match a hand-augmented companion-form Kalman prediction to machine precision; the first-order Markov CMI becomes positive while the second-order CMI vanishes (d-separation); and the tied factor `T` is recovered from observations only, with the analytic CRB matching the empirical MLE scatter for two observation patterns, all sensors vs the odd-indexed half (400 Newton trials each at `M = 2000`; halving the sensors keeps `T` identifiable but inflates the CRB by ~1.3x). Writes `skip_*.dat` (pgfplots data for the paper figure). |
| `exp11_design_gradient.py` | Continuous D-optimal design via nested AD | Same fusion model as exp9, starting from its worst 2-reference placement (loading rows 8.9° apart). One reference row is re-aimed (fixed norm) by Adam ascent on `logdet G(θ)`, whose gradient is a nested automatic derivative through the Fisher metric (`fisher_metric_differentiable`; checked against finite differences to `1.9e-10`). The mean CRB SE drops `0.0372 -> 0.0143`, beating the best exhaustive discrete pair (`0.0150`), and the optimized row rotates to `84.5°` from the fixed reference. |

Output files:

```
experiments/
├── exp{1,2,3,5,6,7,8,9,10,11}_*.py        # run scripts
├── results/exp{1,2,3,5,6,7,8,9,10,11}.json  # numerical results (committed source of truth)
├── results/lds_*.dat            # pgfplots data for the paper LDS figure (exp8)
├── results/calib_*.dat          # pgfplots data (exp9; no longer cited by the paper)
├── make_summary.py           # results/{1,2,3,5}.json -> experiments_summary.tex
└── experiments_summary.pdf   # 4-experiment summary (plots from real trajectories)
```

These are research experiments (not unit tests). For the durable correctness
suite see [`../tests/`](../tests/); for tutorial-scale scripts see
[`../examples/`](../examples/).
