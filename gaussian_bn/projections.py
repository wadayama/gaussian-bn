"""Projection operators for constrained optimization on Gaussian DAG parameters.

Applied inside ``torch.no_grad()`` blocks during projected gradient ascent. They
return new tensors (do not modify inputs in place); the caller uses ``copy_()``
when in-place semantics are desired. Ported verbatim (dtype/device-safe) from the
reference ``gaussian_dag.projections``.
"""

from __future__ import annotations

import torch


def project_frobenius_ball(A: torch.Tensor, P: float) -> torch.Tensor:
    """Project ``A`` onto the Frobenius ball ``{X : ||X||_F^2 <= P}``.

    ``A_proj = A * min(1, sqrt(P) / ||A||_F)`` -- exact Euclidean projection that
    preserves the matrix direction.

    Args:
        A: Real or complex matrix (any shape; Frobenius norm).
        P: Positive power budget.

    Returns:
        New tensor, same shape and dtype as ``A``.
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    norm = torch.linalg.norm(A)
    sqrt_P = torch.sqrt(torch.tensor(P, dtype=norm.dtype, device=norm.device))
    scale = torch.where(norm <= sqrt_P, torch.ones_like(norm), sqrt_P / norm)
    return A * scale


def project_total_power(params: list[torch.Tensor], P: float) -> list[torch.Tensor]:
    """Project a list of matrices onto ``sum_m ||A_m||_F^2 <= P`` (uniform rescale).

    ``A_m_proj = A_m * min(1, sqrt(P) / sqrt(sum_m ||A_m||_F^2))`` -- the Euclidean
    projection of the stacked vector onto a single ball; relative magnitudes are
    preserved.

    Args:
        params: List of real or complex tensors (any shapes).
        P: Positive total power budget.

    Returns:
        List of new tensors (same length and shapes as ``params``).
    """
    if P <= 0:
        raise ValueError(f"Power budget P must be positive, got {P}")
    if len(params) == 0:
        raise ValueError("params must be a non-empty list.")
    total_sq = sum((torch.linalg.norm(p) ** 2) for p in params)
    sqrt_total = torch.sqrt(total_sq)
    sqrt_P = torch.sqrt(torch.tensor(P, dtype=sqrt_total.dtype, device=sqrt_total.device))
    scale = torch.where(sqrt_total <= sqrt_P, torch.ones_like(sqrt_total), sqrt_P / sqrt_total)
    return [p * scale for p in params]
