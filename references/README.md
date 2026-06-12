# References — origins of the K-recursion

A reading collection of the classical papers behind the K-recursion's
covariance map (see [`../MATH.md`](../MATH.md#8-relationship-to-prior-work) and
[`../notes/krecursion_related_work.md`](../notes/krecursion_related_work.md) for
how each relates to this library).

The K-recursion maps the local conditional parameters `{A_{ji}, Σ_j}` of a
linear Gaussian DAG to all node-pair covariance blocks. The identity, the
local-to-global map, and the path/trek view are classical; these are the primary
sources.

> Note: downloaded PDFs are third-party and **git-ignored** (not redistributed
> with the library). Paywalled items are listed link-only.

## Downloaded (in this folder)

| File | Paper | Why it matters |
| --- | --- | --- |
| `Wright_1921_correlation_and_causation_JAgricRes_v20_pp557-585.pdf` | Sewall Wright, *Correlation and Causation*, J. Agricultural Research **20**, 557–585 (1921). **(full Volume 20; Wright's article is pp. 557–585)** | Origin of **path analysis** — covariance as sums over paths/treks. Public domain (Internet Archive). |
| `Geiger_Heckerman_1994_learning_gaussian_networks.pdf` | D. Geiger & D. Heckerman, *Learning Gaussian Networks*, UAI 1994 (arXiv:1302.6808). | Recursive construction of a Gaussian network's covariance/precision from local parameters; builds on Shachter–Kenley. |
| `Sullivant_Talaska_Draisma_2010_trek_separation.pdf` | S. Sullivant, K. Talaska, J. Draisma, *Trek Separation for Gaussian Graphical Models*, Ann. Statist. **38**(3), 1665–1685 (2010) (arXiv:0812.1938). | Modern **trek rule** / trek separation: covariance entries as path polynomials; the parent cross-covariance `K_{ii'}` at merging nodes is a trek contribution. |
| `Kalman_1960_linear_filtering_prediction.pdf` | R. E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, Trans. ASME J. Basic Eng. **82**(1), 35–45 (1960). | The **chain special case**: the self-block recursion `P ← A P A^H + Q` (no cross term). |

## Link-only (paywalled or scripted download blocked)

- **Shachter & Kenley, *Gaussian Influence Diagrams*, Management Science 35(5), 527–550 (1989).** The closest algorithmic prior art (local↔covariance via arc reversal). Paywalled (INFORMS): <https://pubsonline.informs.org/doi/10.1287/mnsc.35.5.527> · <https://ideas.repec.org/a/inm/ormnsc/v35y1989i5p527-550.html>
- **Wright, *The Method of Path Coefficients*, Ann. Math. Statist. 5(3), 161–215 (1934).** Open on Project Euclid (download from the browser): <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-5/issue-3/The-Method-of-Path-Coefficients/10.1214/aoms/1177732676.full>
- **Rauch, Tung & Striebel, *Maximum Likelihood Estimates of Linear Dynamic Systems*, AIAA J. 3(8), 1445–1450 (1965).** The RTS smoother (chain covariance smoothing). Paywalled (AIAA): <https://arc.aiaa.org/doi/abs/10.2514/3.3166>

## Background texts (not papers; library/textbook)

- K. Bollen, *Structural Equations with Latent Variables*, Wiley (1989) — the `K = (I−A)^{-1} Σ (I−A)^{-H}` closed form.
- T. Kailath, *Linear Systems*, Prentice-Hall (1980) — discrete Lyapunov equation.
- D. Koller & N. Friedman, *Probabilistic Graphical Models*, MIT Press (2009) — linear Gaussian networks.
- (MI-optimization context) E. Telatar, *Capacity of Multi-antenna Gaussian Channels*, ETT 10(6), 585–595 (1999); Cover & Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006).

## Suggested reading order

1. **Wright 1921** (path analysis — the intuition: covariance as paths).
2. **Shachter–Kenley 1989** (the local→global covariance map for Gaussian networks; link-only).
3. **Geiger–Heckerman 1994** (the recursive construction; short, 6 pp.).
4. **Sullivant et al. 2010** (the modern trek rule / trek separation — where the merging-node cross terms come from).
5. **Kalman 1960** (the chain special case, to see what the general DAG adds).
