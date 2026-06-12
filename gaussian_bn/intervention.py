"""Interventions (do-operations) on linear Gaussian Bayesian networks.

An intervention is implemented exactly as the notes prescribe: modify the DAG /
edge matrices / noise covariances and re-run the K-recursion. Each operation
returns a NEW :class:`GaussianDAG` (the input is never mutated), so all the
inference machinery applies unchanged to the intervened model.

This release is zero-mean: a hard intervention fixes a node's *mechanism* (it
becomes an independent root with a given covariance and no incoming edges); a
soft intervention replaces a node's incoming edges and/or its noise. Fixing a
node to a deterministic point value (which introduces a nonzero mean) is a
documented follow-up.
"""

from __future__ import annotations

from typing import Sequence

import torch

from typing import Mapping, Sequence

from gaussian_bn.inference import conditional_mean, conditional_mutual_information
from gaussian_bn.model import GaussianDAG, k_full


def _check_node(model: GaussianDAG, node: int) -> None:
    if not 0 <= node < model.M:
        raise ValueError(f"node {node} out of range [0, {model.M}).")


def do_hard(model: GaussianDAG, node: int, *, value: torch.Tensor | None = None,
            cov: torch.Tensor | None = None) -> GaussianDAG:
    """Hard intervention: cut ``node`` from its parents, making it an independent root.

    All incoming edges ``(i, node)`` are deleted, so ``node`` becomes a root.

    - **Mechanism intervention** (default): ``V_node ~ N(0, cov)`` with ``cov``
      defaulting to the existing innovation covariance ``noise[node]``.
    - **Point intervention** ``do(V_node = value)``: pass ``value`` (a length
      ``dims[node]`` vector). The node is fixed deterministically — its offset
      becomes ``value`` and its covariance becomes ``cov`` (default: ``0``, a
      degenerate point mass). The non-zero mean propagates downstream; query the
      non-intervened nodes (a covariance block that includes a deterministic node
      is singular). Requires the model to carry means.

    Returns a new :class:`GaussianDAG`.
    """
    _check_node(model, node)
    if value is None:
        removed = {(i, node): None for i in model.parents[node]}
        out = model.replace_edges(removed)
        return out.replace_noise(node, model.noise[node] if cov is None else cov)
    # point intervention do(V_node = value): deterministic node (degenerate cov).
    d = model.dims[node]
    edges = {k: v for k, v in model.edges.items() if k[1] != node}
    noise = list(model.noise)
    noise[node] = (torch.zeros(d, d, dtype=model.dtype, device=model.device)
                   if cov is None else cov)
    mean = list(model.mean)
    mean[node] = torch.as_tensor(value, dtype=model.dtype, device=model.device).reshape(-1)
    return GaussianDAG(model.dims, edges, noise, dtype=model.dtype,
                       device=model.device, validate=False, mean=mean)


def do_soft(
    model: GaussianDAG, node: int, *,
    edges: dict[int, torch.Tensor] | None = None,
    noise: torch.Tensor | None = None,
) -> GaussianDAG:
    """Soft intervention: replace ``node``'s incoming mechanism.

    Args:
        model: Base model.
        node: Target node.
        edges: Optional ``{parent_i: A'_{node,i}}`` replacements for incoming edges.
        noise: Optional replacement noise covariance ``Sigma'_node``.

    Returns a new :class:`GaussianDAG`.
    """
    _check_node(model, node)
    out = model
    if edges:
        out = out.replace_edges({(i, node): A for i, A in edges.items()})
    if noise is not None:
        out = out.replace_noise(node, noise)
    return out


def do_covariance(model: GaussianDAG, node: int, *, cov: torch.Tensor | None = None) -> torch.Tensor:
    """Full covariance after a hard intervention ``do(node)`` (convenience)."""
    return k_full(do_hard(model, node, cov=cov))


def do_cmi(
    model: GaussianDAG, node: int, A: Sequence[int], B: Sequence[int], C: Sequence[int],
    *, cov: torch.Tensor | None = None, jitter: float = 0.0,
) -> torch.Tensor:
    """Conditional mutual information on the hard-intervened model ``do(node)``."""
    return conditional_mutual_information(do_hard(model, node, cov=cov), A, B, C, jitter=jitter)


def counterfactual(
    model: GaussianDAG,
    evidence: Sequence[int],
    evidence_values: torch.Tensor,
    do: Mapping[int, torch.Tensor],
    query: Sequence[int],
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Counterfactual query ``E[V_query | V_evidence = e ; do(V_j = u_j)]``.

    The Pearl abduction–action–prediction recipe for a linear Gaussian SEM:

    1. **Abduction** — infer the exogenous noise consistent with the evidence,
       from the posterior mean ``v = E[V_all | V_evidence = e]`` (Gaussian
       conditioning) and ``z_k = v_k - c_k - sum_i A_{ki} v_i``.
    2. **Action** — fix the intervened nodes ``V_j = u_j``.
    3. **Prediction** — re-run the structural equations in topological order with
       the *same* abducted noise ``z`` and the intervened values.

    This answers "what would ``V_query`` have been, had ``V_j`` been ``u_j``,
    given that we observed ``V_evidence = e``" — holding the latent randomness of
    the factual world fixed.

    Args:
        model: The Gaussian BN (may carry a mean / offsets).
        evidence: Observed node indices.
        evidence_values: Observed values, stacked over ``evidence`` (length =
            sum of their dims).
        do: Mapping ``{node: value}`` of point interventions.
        query: Node indices whose counterfactual mean is returned.
        jitter: Diagonal shift in the abduction conditioning solve.

    Returns:
        The counterfactual mean of ``query``, stacked (length = sum of their dims).
    """
    all_nodes = list(range(model.M))
    v_post = conditional_mean(model, all_nodes, list(evidence),
                              torch.as_tensor(evidence_values, dtype=model.dtype,
                                              device=model.device).reshape(-1),
                              jitter=jitter)
    vp = {j: v_post[model.slc(j)] for j in range(model.M)}

    # (1) abduction: noise consistent with the evidence
    z = {}
    for k in range(model.M):
        acc = vp[k] - model.mean[k]
        for i in model.parents[k]:
            acc = acc - model.edges[(i, k)] @ vp[i]
        z[k] = acc

    # (2,3) action + prediction: re-run with the same noise and the interventions
    do = {int(j): torch.as_tensor(u, dtype=model.dtype, device=model.device).reshape(-1)
          for j, u in do.items()}
    vcf = {}
    for k in range(model.M):
        if k in do:
            vcf[k] = do[k]
        else:
            acc = model.mean[k] + z[k]
            for i in model.parents[k]:
                acc = acc + model.edges[(i, k)] @ vcf[i]
            vcf[k] = acc
    return torch.cat([vcf[q] for q in query])
