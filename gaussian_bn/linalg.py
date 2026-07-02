"""Numerical linear-algebra helpers for the Gaussian BN K-recursion library.

This module is the single chokepoint for the numerically sensitive operations
used throughout the package, so that hardening choices are made in exactly one
place:

- Hermitian (symmetric) enforcement via ``hermitianize``;
- log-determinants of Hermitian positive-definite matrices via Cholesky
  (``logdet_hpd``), never via ``slogdet`` of a general matrix;
- linear systems via ``solve_psd`` (a Cholesky factor and triangular solves),
  never via an explicit ``torch.linalg.inv``;
- Schur complements that are always re-symmetrized after the subtraction
  (``schur_complement``);
- Cholesky factors with an optional positive-definiteness ``jitter`` floor
  (``cholesky_psd``).

All helpers are dtype/device agnostic: any auxiliary tensor (identity shifts)
is built with the dtype and device of the input, so the same code runs under
``torch.float64`` (the project default) and ``torch.complex128``.
"""

from __future__ import annotations

import torch


def hermitianize(A: torch.Tensor) -> torch.Tensor:
    """Symmetrize a square matrix by ``(A + A^H) / 2``.

    Enforces exact Hermitian structure on tensors that are Hermitian in theory
    but may drift due to floating-point round-off. For real tensors ``A.mH`` is
    the ordinary transpose, so this returns the symmetric part.

    Args:
        A: Square matrix (real or complex), possibly batched in leading dims.

    Returns:
        The Hermitian part of ``A``, same shape and dtype.
    """
    return 0.5 * (A + A.mH)


def _eye_like(A: torch.Tensor) -> torch.Tensor:
    """Identity matrix matching the trailing dimension, dtype, and device of A."""
    d = A.shape[-1]
    return torch.eye(d, dtype=A.dtype, device=A.device)


def logdet_hpd(A: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Cholesky-based log-determinant for Hermitian positive-definite ``A``.

    For Hermitian positive-definite ``A = L L^H`` with lower-triangular ``L`` and
    real positive ``diag(L)``,
        ``log det A = 2 * sum_i log L_ii``.
    The input is symmetrized by ``(A + A^H) / 2`` before factorization.

    Args:
        A: Hermitian PSD matrix (``d x d``, real or complex).
        jitter: If ``> 0``, replace ``A`` by ``A + jitter * I`` before the
            factorization. Use sparingly and report the value in any experiment.

    Returns:
        Real scalar tensor: ``log det A`` (natural log; nats convention).
    """
    A = hermitianize(A)
    if jitter > 0.0:
        A = A + jitter * _eye_like(A)
    L = torch.linalg.cholesky(A)
    return 2.0 * torch.log(torch.diagonal(L).real).sum()


def solve_psd(A: torch.Tensor, B: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """Solve ``(A + jitter*I) X = B`` via a Cholesky factor, never an inverse.

    This is the package-wide replacement for ``torch.linalg.inv(A) @ B``: the
    coefficient matrix is factored once as ``A = L L^H`` (:func:`cholesky_psd`)
    and ``X`` follows from two triangular solves against ``L``
    (``torch.cholesky_solve``). ``A`` must be Hermitian positive definite after
    the optional jitter, as it is in all internal uses.

    Args:
        A: Square coefficient matrix (Hermitian PD in all internal uses).
        B: Right-hand side; shape ``(..., d, k)`` or ``(..., d)``.
        jitter: Optional positive diagonal shift on ``A``.

    Returns:
        ``X`` such that ``(A + jitter*I) X = B``.
    """
    L = cholesky_psd(A, jitter=jitter)
    if B.dim() == A.dim() - 1:                     # vector right-hand side
        return torch.cholesky_solve(B.unsqueeze(-1), L).squeeze(-1)
    return torch.cholesky_solve(B, L)


def schur_complement(
    KAA: torch.Tensor,
    KAB: torch.Tensor,
    KBB: torch.Tensor,
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Hermitian Schur complement ``K_{A|B} = K_AA - K_AB K_BB^{-1} K_BA``.

    Computed as ``hermitianize(KAA - KAB @ solve_psd(KBB, KAB.mH))``. The result
    is always re-symmetrized after the subtraction to remove the small
    non-Hermitian drift that the matrix products introduce; this is the
    conditional covariance ``Cov(V_A | V_B)``.

    Args:
        KAA: ``K_AA`` block (``d_A x d_A``).
        KAB: ``K_AB`` block (``d_A x d_B``); ``K_BA = KAB.mH``.
        KBB: ``K_BB`` block (``d_B x d_B``), Hermitian PD.
        jitter: Optional positive diagonal shift on ``K_BB`` before the solve.

    Returns:
        Hermitian conditional covariance ``K_{A|B}`` (``d_A x d_A``).
    """
    return hermitianize(KAA - KAB @ solve_psd(KBB, KAB.mH, jitter=jitter))


def cholesky_psd(A: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """Lower-triangular Cholesky factor of a Hermitian PD matrix.

    Symmetrizes and optionally jitters ``A`` before factorization, so callers
    (e.g. sampling) get a stable factor even when ``A`` is barely PD.

    Args:
        A: Hermitian PSD matrix (``d x d``).
        jitter: Optional positive diagonal shift added before factorization.

    Returns:
        Lower-triangular ``L`` with ``L L^H = A (+ jitter I)``.
    """
    A = hermitianize(A)
    if jitter > 0.0:
        A = A + jitter * _eye_like(A)
    return torch.linalg.cholesky(A)


def psd_factor(A: torch.Tensor, *, jitter: float = 0.0) -> torch.Tensor:
    """A factor ``L`` with ``L L^H = A`` for Hermitian positive *semi*-definite ``A``.

    Tries the Cholesky factorization first, so for positive-definite input the
    result (and anything seeded on it, e.g. sampling) is bit-identical to
    :func:`cholesky_psd`. Only when the Cholesky fails -- a singular covariance,
    such as the zero matrix of a deterministic node -- does it fall back to a
    symmetric eigendecomposition ``A = V diag(w) V^H`` and return
    ``V diag(sqrt(max(w, 0)))`` (not triangular, which no caller requires).

    Args:
        A: Hermitian PSD matrix (``d x d``), possibly singular.
        jitter: Optional positive diagonal shift added before factorization.

    Returns:
        ``L`` (``d x d``) with ``L L^H = A (+ jitter I)`` up to the clamping of
        tiny negative round-off eigenvalues.
    """
    try:
        return cholesky_psd(A, jitter=jitter)
    except torch.linalg.LinAlgError:
        Ah = hermitianize(A)
        if jitter > 0.0:
            Ah = Ah + jitter * _eye_like(Ah)
        w, V = torch.linalg.eigh(Ah)
        return V * torch.sqrt(torch.clamp(w, min=0.0)).to(V.dtype)
