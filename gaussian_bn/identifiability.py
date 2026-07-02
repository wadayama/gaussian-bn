"""Fisher-based edge identifiability for linear Gaussian Bayesian networks.

The pullback Fisher metric on the observed block,
    G^{(O)}_{ab} = c tr[ K_OO^{-1} (dK_OO/d eta_a) K_OO^{-1} (dK_OO/d eta_b) ],
with the field constant c = 1/2 for real models and c = 1 for circular (proper)
complex models,
measures how sensitively the observed distribution responds to each parameter
direction. A rank deficiency is a non-identifiable direction (e.g. a latent
scale/rotation gauge): the parameter can move without changing the observed
distribution to first order.

:func:`fisher_metric` works on any differentiable map ``eta -> K_OO``;
:func:`edge_fisher` specializes it to a chosen set of edge parameters of a model
and observed node set; :func:`identifiability_report` bundles rank, condition
number, null directions, and per-parameter sensitivity into a dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from gaussian_bn.inference import marginal
from gaussian_bn.linalg import solve_psd
from gaussian_bn.model import GaussianDAG, pack, unpack


def fisher_metric(
    K_of_eta: Callable[[torch.Tensor], torch.Tensor],
    eta0: torch.Tensor,
    *,
    method: Literal["autograd", "fd"] = "autograd",
    fd_eps: float = 1e-6,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pullback Fisher metric ``G``, with its eigenvalues and eigenvectors.

    Args:
        K_of_eta: Differentiable map from a 1-D parameter vector to the observed
            covariance ``K_OO`` (a ``d x d`` Hermitian PD matrix).
        eta0: Point at which to evaluate the metric.
        method: ``"autograd"`` uses ``torch.autograd.functional.jacobian`` in
            forward mode, one Jacobian-vector sweep per parameter (cost ``O(q)``
            in the parameter count ``q``), falling back to reverse mode if an
            operation lacks forward-mode support; ``"fd"`` uses central finite
            differences on ``K_of_eta`` (a robust, derivative-free fallback).
        fd_eps: Step size for the finite-difference Jacobian.
        jitter: Diagonal shift used in the ``K_OO^{-1}`` solves.

    Returns:
        ``(G, eigvals, eigvecs)`` from a symmetric eigendecomposition of ``G``
        (ascending eigenvalues).
    """
    eta0 = torch.as_tensor(eta0).detach()
    q = eta0.numel()
    K0 = K_of_eta(eta0).detach()
    d = K0.shape[-1]

    if method == "autograd":
        def kflat(eta: torch.Tensor) -> torch.Tensor:
            return K_of_eta(eta).reshape(-1)
        try:                                    # one JVP sweep per parameter: O(q)
            J = torch.autograd.functional.jacobian(
                kflat, eta0, strategy="forward-mode", vectorize=True)  # (d*d, q)
        except (RuntimeError, NotImplementedError):
            J = torch.autograd.functional.jacobian(kflat, eta0)       # reverse fallback
        dK = [J[:, a].reshape(d, d) for a in range(q)]
    else:
        dK = []
        for a in range(q):
            e = torch.zeros_like(eta0).reshape(-1)
            e[a] = fd_eps
            e = e.reshape(eta0.shape)
            dKa = (K_of_eta(eta0 + e).detach() - K_of_eta(eta0 - e).detach()) / (2 * fd_eps)
            dK.append(dKa)

    # Precompute K^{-1} dK_a once per parameter, then G_ab = c tr[(K^-1 dK_a)(K^-1 dK_b)]
    # with the field constant c = 1/2 (real) or 1 (circular complex).
    c = 1.0 if K0.is_complex() else 0.5
    KinvdK = [solve_psd(K0, dKa, jitter=jitter) for dKa in dK]
    G = torch.zeros((q, q), dtype=torch.float64, device=K0.device)
    for a in range(q):
        for b in range(q):
            G[a, b] = c * torch.trace(KinvdK[a] @ KinvdK[b]).real
    G = 0.5 * (G + G.mH)
    eigvals, eigvecs = torch.linalg.eigh(G)
    return G, eigvals, eigvecs


def edge_fisher(
    model: GaussianDAG,
    edge_params: Sequence[tuple[int, int]],
    observed: Sequence[int],
    *,
    method: Literal["autograd", "fd"] = "autograd",
    fd_eps: float = 1e-6,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fisher metric over a set of edge parameters for an observed node set.

    The noise covariances are held fixed; only the entries of the edges in
    ``edge_params`` vary. Returns ``(G, eigvals, eigvecs)``.
    """
    eta0, pk = pack(model, free_edges=list(edge_params), noise_param="fixed",
                    requires_grad=False)

    def K_of_eta(eta: torch.Tensor) -> torch.Tensor:
        cur = unpack(eta, pk, model)
        return marginal(cur, observed)

    return fisher_metric(K_of_eta, eta0, method=method, fd_eps=fd_eps, jitter=jitter)


@dataclass
class IdentifiabilityReport:
    """Diagnostic summary of edge identifiability for an observation set.

    Attributes:
        rank: Numerical rank of the Fisher metric ``G``.
        q: Number of edge parameters.
        condition_number: ``lambda_max / lambda_min`` (``inf`` if rank-deficient).
        eigenvalues: Ascending eigenvalues of ``G``.
        null_directions: Eigenvectors whose eigenvalue is below the rank
            tolerance (the non-identifiable / gauge directions).
        per_param_sensitivity: ``diag(G)`` -- the marginal sensitivity of the
            observed distribution to each parameter.
        identifiable: ``True`` iff ``rank == q``.
    """

    rank: int
    q: int
    condition_number: float
    eigenvalues: torch.Tensor
    null_directions: torch.Tensor
    per_param_sensitivity: torch.Tensor
    identifiable: bool


def identifiability_report(
    model: GaussianDAG,
    edge_params: Sequence[tuple[int, int]],
    observed: Sequence[int],
    *,
    method: Literal["autograd", "fd"] = "autograd",
    jitter: float = 0.0,
    rank_tol: float | None = None,
) -> IdentifiabilityReport:
    """Build an :class:`IdentifiabilityReport` from the edge Fisher metric."""
    G, w, U = edge_fisher(model, edge_params, observed, method=method, jitter=jitter)
    q = w.numel()
    wmax = float(w.max()) if q else 0.0
    tol = rank_tol if rank_tol is not None else 1e-8 * max(wmax, 1e-30)
    mask = w > tol
    rank = int(mask.sum())
    wmin_pos = float(w[mask].min()) if rank > 0 else 0.0
    cond = float(wmax / wmin_pos) if rank == q and wmin_pos > 0 else float("inf")
    null_dirs = U[:, ~mask]
    return IdentifiabilityReport(
        rank=rank, q=q, condition_number=cond, eigenvalues=w,
        null_directions=null_dirs, per_param_sensitivity=torch.diagonal(G),
        identifiable=(rank == q),
    )
