"""Parameter estimation for linear Gaussian Bayesian networks.

Three regimes, all built on the K-recursion covariance engine:

* Full observation: the likelihood factorizes and the MLE is closed-form
  node-wise Gaussian regression (:func:`fit_local_regression`,
  :func:`local_mle_from_cov`), optionally ridge-regularized.
* Hidden nodes / partial observation: the observed-block Gaussian negative
  log-likelihood (:func:`gaussian_nll`, :func:`marginal_likelihood`) is a
  differentiable function of the parameters and is maximized either by gradient
  methods (:func:`fit_gradient`) or by EM (:func:`em_fit`), whose E-step uses the
  K-recursion to form posterior moments (:func:`posterior_full_moments`).

Observation patterns are described by :class:`ObsPattern` (a set of observed
nodes and the corresponding data), which lets a single objective handle
missing-data / per-pattern likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch

from gaussian_bn.linalg import hermitianize, logdet_hpd, solve_psd
from gaussian_bn.model import GaussianDAG, NoiseParam, k_full, mean_all, pack, unpack


@dataclass
class ObsPattern:
    """An observation pattern: ``observed`` node indices and data ``Y``.

    ``Y`` has shape ``(N, sum_j dims[j] for j in observed)`` in observed-node
    (stacked) order.
    """

    observed: tuple[int, ...]
    Y: torch.Tensor


def _require_zero_mean(model: GaussianDAG, fn_name: str) -> None:
    """Reject affine models on the zero-mean-only EM path."""
    if model.has_mean:
        raise ValueError(
            f"{fn_name} assumes a zero-mean model, but this model carries a "
            "mean (affine model). Fitting it here would silently absorb the "
            "mean into the covariance estimates. Center the data and fit a "
            "zero-mean model, or use fit_gradient / marginal_likelihood, "
            "which handle the (fixed) model mean."
        )


# --------------------------------------------------------------------------
# Full observation: local Gaussian regression MLE
# --------------------------------------------------------------------------
def local_mle_from_cov(
    model: GaussianDAG, Kmat: torch.Tensor, *, ridge: float = 0.0,
) -> tuple[dict[tuple[int, int], torch.Tensor], list[torch.Tensor]]:
    """Node-wise Gaussian-regression MLE reading blocks from a covariance ``Kmat``.

    ``Kmat`` may be a sample covariance (full observation) or the expected
    sufficient-statistic matrix from an EM E-step. With ``ridge > 0`` the parent
    Gram matrix is shifted, ``B_j = K_{jU}(K_{UU} + ridge I)^{-1}``.

    Returns ``(edges_hat, noise_hat)`` in the :class:`GaussianDAG` constructor
    format (``edges`` keyed ``(i, j)``; ``noise`` a length-``M`` list).
    """
    edges_hat: dict[tuple[int, int], torch.Tensor] = {}
    noise_hat: list[torch.Tensor] = [None] * model.M  # type: ignore[list-item]
    for j in range(model.M):
        sj = model.node_index([j])
        pa = model.parents[j]
        Kjj = Kmat[sj][:, sj]
        if not pa:
            noise_hat[j] = hermitianize(Kjj)
            continue
        ui = model.node_index(pa)
        KjU = Kmat[sj][:, ui]
        KUU = Kmat[ui][:, ui]
        Bj = solve_psd(KUU, KjU.mH, jitter=ridge).mH       # no explicit inverse
        noise_hat[j] = hermitianize(Kjj - Bj @ KjU.mH)
        c = 0
        for i in pa:
            di = model.dims[i]
            edges_hat[(i, j)] = Bj[:, c:c + di]
            c += di
    return edges_hat, noise_hat


def fit_local_regression(
    model: GaussianDAG, X: torch.Tensor, *, ridge: float = 0.0, fit_mean: bool = False,
):
    """Full-observation MLE from a data matrix ``X`` of shape ``(N, D)``.

    With ``fit_mean=False`` (default) the model is assumed zero-mean and the
    result is ``(edges_hat, noise_hat)`` (backward-compatible). With
    ``fit_mean=True`` the affine model ``V_j = c_j + sum_i A_{ji} V_i + Z_j`` is
    fitted: the data is centered, the edges/noise are estimated from the centered
    covariance, the node offsets ``c_j = x̄_j - B_j x̄_{U_j}`` are recovered, and
    the result is ``(edges_hat, noise_hat, mean_hat)``.
    """
    N = X.shape[0]
    if not fit_mean:
        return local_mle_from_cov(model, (X.mH @ X) / N, ridge=ridge)
    xbar = X.mean(0)
    Xc = X - xbar
    edges_hat, noise_hat = local_mle_from_cov(model, (Xc.mH @ Xc) / N, ridge=ridge)
    mean_hat: list[torch.Tensor] = []
    for j in range(model.M):
        xj = xbar[model.slc(j)]
        pa = model.parents[j]
        if not pa:
            mean_hat.append(xj.clone())                  # root: c_j = E[V_j]
        else:
            Bj = torch.cat([edges_hat[(i, j)] for i in pa], dim=1)
            xU = torch.cat([xbar[model.slc(i)] for i in pa])
            mean_hat.append(xj - Bj @ xU)
    return edges_hat, noise_hat, mean_hat


# --------------------------------------------------------------------------
# Marginal likelihood (hidden nodes / missing data)
# --------------------------------------------------------------------------
def gaussian_nll(K_OO: torch.Tensor, S: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """Per-sample Gaussian NLL up to a constant: ``logdet K_OO + tr(K_OO^{-1} S)``.

    ``S`` is the observed sample covariance. Differentiable through ``K_OO``.
    """
    return logdet_hpd(K_OO, jitter) + torch.trace(solve_psd(K_OO, S, jitter=jitter))


def _observed_block(model: GaussianDAG, observed: Sequence[int],
                    Kf: torch.Tensor | None = None) -> torch.Tensor:
    Kf = k_full(model) if Kf is None else Kf
    oi = model.node_index(observed)
    return Kf[oi][:, oi]


def marginal_likelihood(model: GaussianDAG, patterns: Sequence[ObsPattern],
                        *, jitter: float = 0.0) -> torch.Tensor:
    """Average observed-data NLL over (possibly several) observation patterns.

    ``mean_n [ logdet K_{O_n O_n} + y_n^H K_{O_n O_n}^{-1} y_n ]``, grouped by
    pattern. Reduces to :func:`gaussian_nll` for a single pattern with ``S`` its
    sample covariance. Differentiable through the model parameters.
    """
    Kf = k_full(model)
    m = mean_all(model) if model.has_mean else None
    total = torch.zeros((), dtype=Kf.real.dtype, device=model.device)
    count = 0
    for pat in patterns:
        K_OO = _observed_block(model, pat.observed, Kf)
        Y = pat.Y
        if m is not None:                              # center by the observed mean
            Y = Y - m[model.node_index(pat.observed)]
        n = Y.shape[0]
        ld = logdet_hpd(K_OO, jitter)
        quad = (Y.conj() * solve_psd(K_OO, Y.mH, jitter=jitter).mH).real.sum()
        total = total + n * ld + quad
        count += n
    return total / count


def posterior_full_moments(model: GaussianDAG, observed: Sequence[int], Y: torch.Tensor,
                           Kf: torch.Tensor | None = None, *, jitter: float = 0.0) -> torch.Tensor:
    """EM E-step expected sufficient statistic ``\\tilde K`` (``D x D``).

    ``\\tilde K = Cov(V_all | V_O) + (1/N) sum_n m_n m_n^H`` with
    ``m_n = K_{all,O} K_OO^{-1} y_n``. Zero-mean models only (the conditioning
    above has no mean shift); affine models are rejected.

    Raises:
        ValueError: If ``model`` carries a mean (affine model).
    """
    _require_zero_mean(model, "posterior_full_moments")
    Kf = k_full(model) if Kf is None else Kf
    oi = model.node_index(observed)
    K_OO = Kf[oi][:, oi]
    Kall_O = Kf[:, oi]
    W = solve_psd(K_OO, Kall_O.mH, jitter=jitter)     # |O| x D = K_OO^{-1} K_{O,all}
    Means = Y @ W                                     # N x D
    N = Y.shape[0]
    C = Kf - Kall_O @ W
    return hermitianize(C + (Means.mH @ Means) / N)


def em_fit(model: GaussianDAG, X: torch.Tensor, observed: Sequence[int], num_iters: int,
           *, ridge: float = 0.0, jitter: float = 0.0) -> tuple[GaussianDAG, list[float]]:
    """EM for a hidden-node BN (known structure). Zero-mean models only.

    Returns ``(fitted_model, nll_history)`` where ``nll_history[t]`` is the
    observed-data NLL at the parameters before the ``t``-th update.

    Raises:
        ValueError: If ``model`` carries a mean (affine model). The E/M steps
            here are zero-mean; an unguarded affine model would silently absorb
            the mean into the covariance estimates and bias every parameter.
            Center the data and fit a zero-mean model, or use
            :func:`fit_gradient`, whose :func:`marginal_likelihood` objective
            handles the (fixed) model mean.
    """
    _require_zero_mean(model, "em_fit")
    oi = model.node_index(observed)
    Y = X[:, oi]
    S = (Y.mH @ Y) / X.shape[0]
    history: list[float] = []
    for _ in range(num_iters):
        Kf = k_full(model)
        history.append(float(gaussian_nll(Kf[oi][:, oi], S, jitter=jitter)))
        Ktil = posterior_full_moments(model, observed, Y, Kf, jitter=jitter)   # E-step
        eh, nh = local_mle_from_cov(model, Ktil, ridge=ridge)                  # M-step
        # PD floor on noise for stability
        nh = [hermitianize(s) + jitter * torch.eye(s.shape[-1], dtype=model.dtype,
                                                   device=model.device) for s in nh]
        model = GaussianDAG(model.dims, eh, nh, dtype=model.dtype,
                            device=model.device, validate=False)
    return model, history


def fit_gradient(
    model: GaussianDAG, patterns: Sequence[ObsPattern], *,
    optimizer: Literal["adam", "lbfgs"] = "adam",
    lr: float = 1e-2, num_iters: int = 200,
    free_edges: Sequence[tuple[int, int]] | None = None,
    noise_param: NoiseParam = "chol",
    regularizer: Callable[[GaussianDAG], torch.Tensor] | None = None,
    jitter: float = 0.0,
) -> tuple[GaussianDAG, list[float]]:
    """Maximize the marginal likelihood by autograd gradient descent on the NLL.

    Packs ``model`` into unconstrained leaf parameters (free edges + a
    PD-preserving noise parametrization), runs Adam or LBFGS on
    :func:`marginal_likelihood` (plus an optional ``regularizer(model)``), and
    returns the fitted :class:`GaussianDAG` and the NLL history.
    """
    eta, pk = pack(model, free_edges=free_edges, noise_param=noise_param,
                   requires_grad=True)

    def objective() -> torch.Tensor:
        cur = unpack(eta, pk, model)
        loss = marginal_likelihood(cur, patterns, jitter=jitter)
        if regularizer is not None:
            loss = loss + regularizer(cur)
        return loss

    history = _optimize_params([eta], objective, optimizer, lr, num_iters)
    return unpack(eta.detach(), pk, model), history


def _optimize_params(
    params: list[torch.Tensor],
    objective: Callable[[], torch.Tensor],
    optimizer: Literal["adam", "lbfgs"],
    lr: float,
    num_iters: int,
) -> list[float]:
    """Run Adam or LBFGS on ``objective`` over the leaf ``params``; return the loss history."""
    history: list[float] = []
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
        for _ in range(num_iters):
            opt.zero_grad()
            loss = objective()
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
    else:  # lbfgs
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=num_iters,
                                tolerance_grad=1e-14, tolerance_change=1e-16,
                                line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = objective()
            loss.backward()
            history.append(float(loss.detach()))
            return loss

        opt.step(closure)
    return history


def fit_gradient_custom(
    params: list[torch.Tensor],
    build_model: Callable[[], GaussianDAG],
    patterns: Sequence[ObsPattern],
    *,
    optimizer: Literal["adam", "lbfgs"] = "adam",
    lr: float = 1e-2,
    num_iters: int = 200,
    regularizer: Callable[[GaussianDAG], torch.Tensor] | None = None,
    jitter: float = 0.0,
) -> list[float]:
    """Train a custom (shared / factored / constrained) edge parametrization.

    The general-purpose companion to :func:`fit_gradient`. Use it whenever the
    edge matrices are *not* free and independent — e.g. a relay factor shared
    across a node's outgoing edges (``A_{ji} = H_{ji} F_i``), a low-rank edge
    (``A = U V^H``), tied blocks, or any other constraint. Because the
    K-recursion is differentiable, autograd back-propagates through *any*
    construction of the edge matrices.

    Args:
        params: Leaf tensors to optimize (each with ``requires_grad=True``) — the
            shared factors, noise parameters, etc. They are updated **in place**.
        build_model: A zero-argument closure that constructs a
            :class:`GaussianDAG` from the current ``params`` (typically with
            ``validate=False``).
        patterns: Observation patterns whose marginal likelihood is minimized.
        optimizer: ``"adam"`` or ``"lbfgs"``.
        lr: Learning rate / LBFGS step.
        num_iters: Number of iterations (Adam) or LBFGS ``max_iter``.
        regularizer: Optional ``regularizer(model) -> scalar`` added to the NLL.
        jitter: Diagonal shift in the likelihood's solves.

    Returns:
        The NLL history. After the call, ``params`` hold the fitted values and
        ``build_model()`` returns the fitted model.

    Raises:
        ValueError: If any entry of ``params`` does not require grad.
    """
    for p in params:
        if not p.requires_grad:
            raise ValueError("All entries of `params` must have requires_grad=True.")

    def objective() -> torch.Tensor:
        model = build_model()
        loss = marginal_likelihood(model, patterns, jitter=jitter)
        if regularizer is not None:
            loss = loss + regularizer(model)
        return loss

    return _optimize_params(params, objective, optimizer, lr, num_iters)
