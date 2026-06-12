"""Projected gradient ascent and natural-gradient ascent for Gaussian DAG objectives.

``pga_ascent`` is the topology-agnostic Euclidean projected-gradient outer loop
ported from the reference ``gaussian_dag.optimize``. ``natural_gradient_step`` and
``natural_gradient_ascent`` precondition the gradient by the inverse of a metric
(e.g. the pullback Fisher metric), giving the Riemannian update
``eta <- eta + alpha G(eta)^{-1} grad f(eta)`` -- a steepest ascent that is
invariant to smooth reparametrizations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from gaussian_bn.linalg import solve_psd


def pga_ascent(
    compute_objective: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None] | None = None,
) -> list[float]:
    """Run projected gradient ASCENT on a scalar objective.

    At each iteration: zero grads, evaluate ``I = compute_objective()`` and
    ``I.backward()`` (recording ``I.item()``), then under ``torch.no_grad()``
    update ``p += step_size * p.grad`` and optionally call ``projector(params)``
    (which mutates ``params`` in place onto the feasible set).

    Args:
        compute_objective: Closure constructing the autograd graph and returning
            the scalar objective tensor.
        params: Leaf tensors with ``requires_grad=True``.
        step_size: Constant positive step size.
        num_iters: Number of iterations (> 0).
        projector: Optional in-place projector applied after each step.

    Returns:
        ``history[t]`` = objective evaluated at the pre-update parameters of
        iteration ``t``.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    for p in params:
        if not p.requires_grad:
            raise ValueError("All entries of `params` must have requires_grad=True.")

    history: list[float] = []
    for _ in range(num_iters):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        I = compute_objective()
        I.backward()
        history.append(I.item())
        with torch.no_grad():
            for p in params:
                p.add_(step_size * p.grad)
            if projector is not None:
                projector(params)
    return history


def natural_gradient_step(
    eta: torch.Tensor, grad: torch.Tensor, metric: torch.Tensor,
    *, step_size: float, jitter: float = 0.0,
) -> torch.Tensor:
    """One natural-gradient ascent step ``eta + step_size * metric^{-1} grad``.

    Uses a linear solve (never an explicit inverse).
    """
    return eta + step_size * solve_psd(metric, grad, jitter=jitter)


def natural_gradient_ascent(
    compute_objective: Callable[[torch.Tensor], torch.Tensor],
    metric_of_eta: Callable[[torch.Tensor], torch.Tensor],
    eta0: torch.Tensor,
    *,
    step_size: float,
    num_iters: int,
    jitter: float = 0.0,
) -> tuple[torch.Tensor, list[float]]:
    """Natural-gradient ascent on ``compute_objective`` preconditioned by a metric.

    At each step the Euclidean gradient of ``compute_objective`` is obtained by
    autograd, then preconditioned by ``metric_of_eta(eta)^{-1}``.

    Args:
        compute_objective: Differentiable scalar objective of ``eta``.
        metric_of_eta: Returns the (Hermitian PD) metric at ``eta``.
        eta0: Initial parameter vector.
        step_size: Positive step size.
        num_iters: Number of iterations (> 0).
        jitter: Diagonal shift for the metric solve.

    Returns:
        ``(eta, history)`` -- the final parameters and the objective history.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    eta = eta0.detach().clone()
    history: list[float] = []
    for _ in range(num_iters):
        e = eta.detach().clone().requires_grad_(True)
        obj = compute_objective(e)
        history.append(float(obj.detach()))
        (grad,) = torch.autograd.grad(obj, e)
        metric = metric_of_eta(eta).detach()
        eta = natural_gradient_step(eta, grad.detach(), metric,
                                    step_size=step_size, jitter=jitter)
    return eta, history
