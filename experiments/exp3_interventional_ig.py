"""Experiment 3: interventional information geometry.

Covers the inference-note Theme D. A confounded triangle
    0 -> 1 (a),  0 -> 2 (b),  1 -> 2 (c)
has node 0 as a hidden common cause of nodes 1 and 2, plus a *direct* causal
edge 1 -> 2 of strength c. The observational association I(V1; V2) mixes the
confounding path (through 0) with the direct causal path. A hard intervention
do(V1) cuts node 1 from its parent, so the post-intervention association
I^{do}(V1; V2) reflects ONLY the direct causal effect; the gap
    Delta I = I(V1;V2) - I^{do}(V1;V2)
is the confounding (non-causal) component.

We sweep the direct-edge strength c and record both quantities, plus the
covariance Cov(V1,V2) before/after the intervention.

Results -> results/exp3.json.  Run: uv run python experiments/exp3_interventional_ig.py
"""

from __future__ import annotations

import json
import os

import torch

import gaussian_bn as gbn

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results", "exp3.json")
SEED = 20260610

A, B = 1.0, 1.2          # confounder -> 1, confounder -> 2 (fixed)
S0, S1, S2 = 1.0, 0.4, 0.5


def triangle(c):
    """Confounded triangle 0->1, 0->2, 1->2 with direct edge strength c."""
    dims = [1, 1, 1]
    edges = {(0, 1): [[A]], (0, 2): [[B]], (1, 2): [[c]]}
    noise = [[[S0]], [[S1]], [[S2]]]
    return gbn.GaussianDAG(dims, edges, noise)


def cov12(model):
    Kf = gbn.k_full(model)
    return float(Kf[model.slc(1), model.slc(2)].item())


def main():
    cs = [round(0.15 * k, 4) for k in range(0, 11)]   # c = 0.0 .. 1.5
    sweep = []
    for c in cs:
        m = triangle(c)
        I_obs = float(gbn.mutual_information(m, [1], [2]))
        m_do = gbn.do_hard(m, 1)                       # cut node 1 from confounder 0
        I_do = float(gbn.mutual_information(m_do, [1], [2]))
        sweep.append({
            "c": c,
            "I_obs": I_obs,
            "I_do": I_do,
            "delta_I_confounding": I_obs - I_do,
            "cov12_obs": cov12(m),
            "cov12_do": cov12(m_do),
        })

    # representative slices
    pure_confounding = sweep[0]                         # c = 0
    mixed = next(s for s in sweep if abs(s["c"] - 0.9) < 1e-9)

    results = {
        "seed": SEED,
        "params": {"a_0to1": A, "b_0to2": B, "noise": [S0, S1, S2]},
        "c_values": cs,
        "sweep": sweep,
        "pure_confounding_c0": pure_confounding,
        "mixed_c0p9": mixed,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("   c     I_obs     I_do    confounding(I_obs-I_do)  cov12_obs  cov12_do")
    for s in sweep:
        print(f" {s['c']:.2f}  {s['I_obs']:.4f}  {s['I_do']:.4f}      {s['delta_I_confounding']:.4f}"
              f"            {s['cov12_obs']:+.4f}   {s['cov12_do']:+.4f}")
    print(f"\nc=0 (pure confounding): I_obs={pure_confounding['I_obs']:.4f}, I_do={pure_confounding['I_do']:.2e}")
    print(f"c=0.9 (mixed): I_obs={mixed['I_obs']:.4f}, I_do={mixed['I_do']:.4f}, "
          f"confounding={mixed['delta_I_confounding']:.4f}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
