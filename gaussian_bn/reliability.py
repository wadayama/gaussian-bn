"""Cramér–Rao bounds and estimator reliability via the Slepian–Bangs Fisher.

For a zero-mean Gaussian observation `V_O ~ N(0, K_OO(θ))`, the Slepian–Bangs
formula gives the per-sample Fisher information

    F_{ab} = ½ tr[ K_OO^{-1} (∂K_OO/∂θ_a) K_OO^{-1} (∂K_OO/∂θ_b) ],

(the ½ becomes 1 for circular complex models), which is exactly the pullback
metric computed by
:func:`gaussian_bn.identifiability.fisher_metric`. For `N` i.i.d. observations
the total information is `N · F`, and the **Cramér–Rao bound** is its inverse:

    Cov(θ̂) ⪰ (N F)^{-1}.

Any unbiased estimator's covariance is lower-bounded by `(N F)^{-1}`; the
maximum-likelihood / EM estimators attain it asymptotically, so the diagonal
gives asymptotic standard errors. When `F` is rank-deficient — a latent gauge
in a partially observed BN — the bound is unbounded along the null directions,
and the affected parameters are reported as non-identifiable (`SE = ∞`).

This module produces `K_OO(θ)` from the model via the K-recursion, so the Fisher
information is computed analytically (no sampling) — the Slepian–Bangs route in
action. The parameter vector `θ` may contain edge entries and, optionally, a
positive-definite-preserving noise parametrization, via the same `pack`/`unpack`
machinery used for training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from gaussian_bn.identifiability import fisher_metric
from gaussian_bn.inference import marginal
from gaussian_bn.linalg import solve_psd
from gaussian_bn.model import GaussianDAG, NoiseParam, mean_all, pack, unpack


def _labels(pk) -> list[str]:
    """Human-readable parameter names matching the packed vector layout."""
    labels: list[str] = []
    for kind, key, sl in pk.slices:
        n = sl.stop - sl.start
        if kind == "edge":
            i, j = key
            _, c = pk.edge_shapes[key]
            for idx in range(n):
                rr, cc = divmod(idx, c)
                labels.append(f"A[{i}->{j}][{rr},{cc}]")
        elif kind == "noise_logdiag":
            for k in range(n):
                labels.append(f"logSig[{key}][{k}]")
        elif kind == "noise_chol_diag":
            for k in range(n):
                labels.append(f"cholDiag[{key}][{k}]")
        elif kind == "noise_chol_strict":
            for k in range(n):
                labels.append(f"cholStrict[{key}][{k}]")
        else:  # pragma: no cover - defensive
            labels.extend([f"{kind}[{key}][{k}]" for k in range(n)])
    return labels


def _mean_fisher_term(model, observed, eta0, pk, K0, method, fd_eps, jitter):
    """Slepian–Bangs mean term ``(∂m_O/∂θ)^H K_OO^{-1} (∂m_O/∂θ)`` (q x q)."""
    oi = model.node_index(observed)

    def m_of_eta(eta: torch.Tensor) -> torch.Tensor:
        return mean_all(unpack(eta, pk, model))[oi]

    if method == "autograd":
        Jm = torch.autograd.functional.jacobian(m_of_eta, eta0)        # (|O|, q)
    else:
        cols = []
        for a in range(eta0.numel()):
            e = torch.zeros_like(eta0).reshape(-1)
            e[a] = fd_eps
            e = e.reshape(eta0.shape)
            cols.append((m_of_eta(eta0 + e).detach() - m_of_eta(eta0 - e).detach()) / (2 * fd_eps))
        Jm = torch.stack(cols, dim=1)
    return (Jm.mH @ solve_psd(K0, Jm, jitter=jitter)).real


def parameter_fisher(
    model: GaussianDAG,
    observed: Sequence[int],
    *,
    free_edges: Sequence[tuple[int, int]] | None = None,
    noise_param: NoiseParam = "fixed",
    include_mean: bool = False,
    method: Literal["autograd", "fd"] = "autograd",
    fd_eps: float = 1e-6,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Per-sample Slepian–Bangs Fisher information over the chosen parameters.

    Computes the covariance term
    ``½ tr[K_OO^{-1} ∂_a K_OO K_OO^{-1} ∂_b K_OO]``. With ``include_mean=True``
    (and a model carrying offsets) it also adds the mean term
    ``(∂m_O/∂θ_a)^H K_OO^{-1} (∂m_O/∂θ_b)`` — the full Slepian–Bangs information.

    Args:
        model: The Gaussian BN (its current parameter values are the point at
            which the Fisher information is evaluated).
        observed: Observed node indices `O`; the Fisher is computed from `K_OO`.
        free_edges: Edge keys to include in `θ` (default: all edges).
        noise_param: ``"fixed"`` excludes noise (edges only); ``"chol"`` /
            ``"logdiag"`` add a PD-preserving noise parametrization (the Fisher
            and CRB are then expressed in those reparametrized noise
            coordinates).
        include_mean: Add the mean term (default ``False``). This is appropriate
            only when the node offsets are *known* — then the parameters carry
            extra information through the mean and the bound tightens. If the
            offsets are themselves estimated (the usual case), leave this
            ``False``: by the mean/covariance orthogonality of Gaussian MLE the
            covariance term alone is the relevant bound.
        method: ``"autograd"`` (Jacobian, `O(q)`) or ``"fd"`` (finite difference).
        fd_eps: Finite-difference step.
        jitter: Diagonal shift in the `K_OO^{-1}` solves.

    Returns:
        ``(F, eigvals, eigvecs, labels)`` — the per-sample FIM and its symmetric
        eigendecomposition, plus parameter labels matching the packed layout.
    """
    eta0, pk = pack(model, free_edges=free_edges, noise_param=noise_param,
                    requires_grad=False)

    def K_of_eta(eta: torch.Tensor) -> torch.Tensor:
        return marginal(unpack(eta, pk, model), observed)

    F, w, U = fisher_metric(K_of_eta, eta0, method=method, fd_eps=fd_eps, jitter=jitter)
    if include_mean and model.has_mean:
        K0 = K_of_eta(eta0).detach()
        F = F + _mean_fisher_term(model, observed, eta0, pk, K0, method, fd_eps, jitter)
        w, U = torch.linalg.eigh(F)
    return F, w, U, _labels(pk)


def _crb_from_fisher(F: torch.Tensor, w: torch.Tensor, U: torch.Tensor, N: int,
                     rank_tol: float | None):
    """CRB matrix, standard errors, rank, and the non-identifiable mask.

    Uses the Moore–Penrose pseudo-inverse of `N F` on the identifiable subspace;
    a parameter whose unit vector has a component in `null(F)` is unidentifiable
    and receives `SE = ∞`.
    """
    q = w.numel()
    wmax = float(w.max()) if q else 0.0
    tol = rank_tol if rank_tol is not None else 1e-8 * max(wmax, 1e-30)
    pos = w > tol
    rank = int(pos.sum())

    inv_w = torch.where(pos, 1.0 / (N * w), torch.zeros_like(w))
    crb = (U * inv_w) @ U.mH                      # U diag(inv_w) U^H = pinv(N F)
    crb = 0.5 * (crb + crb.mH)

    U_null = U[:, ~pos]                           # null-space basis (q x (q-rank))
    if U_null.shape[1] > 0:
        null_proj = torch.linalg.norm(U_null, dim=1)   # ||P_null e_a|| per parameter
    else:
        null_proj = torch.zeros(q, dtype=crb.real.dtype, device=crb.device)
    unident = null_proj > 1e-6

    se = torch.sqrt(torch.clamp(torch.diagonal(crb).real, min=0.0))
    se = torch.where(unident, torch.full_like(se, float("inf")), se)
    return crb.real, se, rank, unident, U_null


def crb(
    model: GaussianDAG,
    observed: Sequence[int],
    N: int,
    *,
    free_edges: Sequence[tuple[int, int]] | None = None,
    noise_param: NoiseParam = "fixed",
    include_mean: bool = False,
    method: Literal["autograd", "fd"] = "autograd",
    fd_eps: float = 1e-6,
    jitter: float = 0.0,
    rank_tol: float | None = None,
) -> torch.Tensor:
    """Cramér–Rao bound matrix `(N F)^{-1}` for the chosen parameters.

    Returns the `q × q` CRB; for a rank-deficient Fisher it is the Moore–Penrose
    pseudo-inverse of `N F` (the minimum-variance bound on the identifiable
    subspace). See :func:`crb_report` for per-parameter standard errors and
    identifiability flags.
    """
    F, w, U, _ = parameter_fisher(model, observed, free_edges=free_edges,
                                  noise_param=noise_param, include_mean=include_mean,
                                  method=method, fd_eps=fd_eps, jitter=jitter)
    crb_mat, _, _, _, _ = _crb_from_fisher(F, w, U, N, rank_tol)
    return crb_mat


@dataclass
class CRBReport:
    """Estimator-reliability report from the Slepian–Bangs Fisher / Cramér–Rao bound.

    Attributes:
        labels: Parameter names matching the packed `θ` vector.
        N: Number of i.i.d. observations the bound is computed for.
        q: Number of parameters; `rank`: numerical rank of `F`.
        fisher: Per-sample Fisher information `F` (`q × q`).
        crb: Cramér–Rao bound `(N F)^{-1}` (pseudo-inverse if rank-deficient).
        standard_errors: `sqrt(diag(crb))`; `inf` for non-identifiable parameters.
        identifiable: `True` iff `rank == q` (every parameter has a finite bound).
        unidentifiable_mask: per-parameter boolean (`True` where `SE = ∞`).
        null_directions: gauge directions (columns), in parameter space.
        confidence: two-sided confidence level used for the intervals.
        theta_hat: the point estimate (the model's packed parameters).
        ci_half_widths: `z · standard_errors` (half-widths of the CIs).
        confidence_intervals: `[θ̂ − hw, θ̂ + hw]` stacked as `(q, 2)`.
    """

    labels: list[str]
    N: int
    q: int
    rank: int
    fisher: torch.Tensor
    crb: torch.Tensor
    standard_errors: torch.Tensor
    identifiable: bool
    unidentifiable_mask: torch.Tensor
    null_directions: torch.Tensor
    confidence: float
    theta_hat: torch.Tensor
    ci_half_widths: torch.Tensor
    confidence_intervals: torch.Tensor

    def summary(self) -> str:
        """A formatted per-parameter table (estimate, SE, confidence interval)."""
        lines = [
            f"CRB reliability report  (N={self.N}, rank {self.rank}/{self.q}, "
            f"identifiable={self.identifiable}, {int(self.confidence * 100)}% CI)",
            f"{'parameter':22s} {'estimate':>11s} {'std.err':>11s} {'CI lower':>11s} {'CI upper':>11s}",
        ]
        for a, lab in enumerate(self.labels):
            se = float(self.standard_errors[a])
            th = float(self.theta_hat[a])
            if math.isinf(se):
                lines.append(f"{lab:22s} {th:11.5f} {'inf':>11s} "
                             f"{'-inf':>11s} {'+inf':>11s}  (non-identifiable)")
            else:
                lo = float(self.confidence_intervals[a, 0])
                hi = float(self.confidence_intervals[a, 1])
                lines.append(f"{lab:22s} {th:11.5f} {se:11.5f} {lo:11.5f} {hi:11.5f}")
        return "\n".join(lines)


def crb_report(
    model: GaussianDAG,
    observed: Sequence[int],
    N: int,
    *,
    free_edges: Sequence[tuple[int, int]] | None = None,
    noise_param: NoiseParam = "fixed",
    include_mean: bool = False,
    confidence: float = 0.95,
    theta_hat: torch.Tensor | None = None,
    method: Literal["autograd", "fd"] = "autograd",
    fd_eps: float = 1e-6,
    jitter: float = 0.0,
    rank_tol: float | None = None,
) -> CRBReport:
    """Build a :class:`CRBReport`: standard errors, confidence intervals, and
    identifiability flags for the parameters of ``model`` estimated from ``N``
    observations of ``observed``.

    The point estimate defaults to the model's own parameter values; pass
    ``theta_hat`` to centre the confidence intervals elsewhere. The intervals are
    the asymptotic normal CIs ``θ̂ ± z · SE`` with ``z = Φ^{-1}((1+confidence)/2)``.
    The full Slepian–Bangs information (including the mean term) is used whenever
    the model has a non-zero mean (override with ``include_mean``).
    """
    F, w, U, labels = parameter_fisher(
        model, observed, free_edges=free_edges, noise_param=noise_param,
        include_mean=include_mean, method=method, fd_eps=fd_eps, jitter=jitter)
    crb_mat, se, rank, unident, U_null = _crb_from_fisher(F, w, U, N, rank_tol)

    if theta_hat is None:
        eta0, _ = pack(model, free_edges=free_edges, noise_param=noise_param,
                       requires_grad=False)
        th = eta0.real
    else:
        th = torch.as_tensor(theta_hat).real
    z = math.sqrt(2.0) * float(torch.special.erfinv(torch.tensor(confidence)))
    hw = z * se
    ci = torch.stack([th - hw, th + hw], dim=1)

    return CRBReport(
        labels=labels, N=N, q=w.numel(), rank=rank,
        fisher=F, crb=crb_mat, standard_errors=se,
        identifiable=(rank == w.numel()), unidentifiable_mask=unident,
        null_directions=U_null, confidence=confidence, theta_hat=th,
        ci_half_widths=hw, confidence_intervals=ci,
    )
