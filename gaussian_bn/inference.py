"""Inference queries on a linear Gaussian BN: marginals, conditioning, (C)MI, sampling.

All covariance queries operate on the full covariance ``Kf`` produced by
:func:`gaussian_bn.model.k_full`. Each query accepts ``Kf=None`` and will build
it from the model on demand, so callers can either pass a cached ``Kf`` (cheap,
matches the experiment scripts) or omit it for convenience.

Conditioning uses the Schur complement (re-symmetrized) and a linear solve, never
an explicit inverse:
    E[V_A | V_B = b] = K_AB K_BB^{-1} b
    Cov(V_A | V_B)   = K_AA - K_AB K_BB^{-1} K_BA.
Mutual information and conditional mutual information use the log-det form, with
the real-Gaussian 1/2 convention (nats):
    I(V_A; V_B)        = 1/2 [logdet K_AA   - logdet K_{A|B}]
    I(V_A; V_B | V_C)  = 1/2 [logdet K_{A|C} - logdet K_{A|BC}].
"""

from __future__ import annotations

from typing import Sequence

import torch

from gaussian_bn.linalg import logdet_hpd, psd_factor, schur_complement, solve_psd
from gaussian_bn.model import GaussianDAG, k_full, mean_all


def _resolve(model: GaussianDAG, Kf: torch.Tensor | None) -> torch.Tensor:
    return k_full(model) if Kf is None else Kf


def _sub(model: GaussianDAG, Kf: torch.Tensor,
         rows: Sequence[int], cols: Sequence[int]) -> torch.Tensor:
    ri = model.node_index(rows)
    ci = model.node_index(cols)
    return Kf[ri][:, ci]


def marginal(model: GaussianDAG, nodes: Sequence[int],
             Kf: torch.Tensor | None = None) -> torch.Tensor:
    """Marginal covariance ``K_AA`` of a node set ``A = nodes``."""
    Kf = _resolve(model, Kf)
    return _sub(model, Kf, nodes, nodes)


def conditional_covariance(model: GaussianDAG, A: Sequence[int], B: Sequence[int],
                           Kf: torch.Tensor | None = None, *, jitter: float = 0.0) -> torch.Tensor:
    """``Cov(V_A | V_B)`` via the (re-symmetrized) Schur complement."""
    Kf = _resolve(model, Kf)
    return schur_complement(_sub(model, Kf, A, A), _sub(model, Kf, A, B),
                            _sub(model, Kf, B, B), jitter=jitter)


def conditional_mean(model: GaussianDAG, A: Sequence[int], B: Sequence[int], b: torch.Tensor,
                     Kf: torch.Tensor | None = None, *, jitter: float = 0.0) -> torch.Tensor:
    """Posterior mean ``E[V_A | V_B = b]``.

    Zero-mean: ``K_AB K_BB^{-1} b``. With a non-zero model mean it is the
    mean-shifted Gaussian conditioning ``m_A + K_AB K_BB^{-1} (b - m_B)``.
    """
    Kf = _resolve(model, Kf)
    KAB = _sub(model, Kf, A, B)
    KBB = _sub(model, Kf, B, B)
    b = torch.as_tensor(b, dtype=model.dtype, device=model.device)
    if not model.has_mean:
        return KAB @ solve_psd(KBB, b, jitter=jitter)
    m = mean_all(model)
    mA = m[model.node_index(A)]
    mB = m[model.node_index(B)]
    return mA + KAB @ solve_psd(KBB, b - mB, jitter=jitter)


def mutual_information(model: GaussianDAG, A: Sequence[int], B: Sequence[int],
                      Kf: torch.Tensor | None = None, *, jitter: float = 0.0) -> torch.Tensor:
    """``I(V_A; V_B) = 1/2 [logdet K_AA - logdet K_{A|B}]`` (nats)."""
    Kf = _resolve(model, Kf)
    KAA = _sub(model, Kf, A, A)
    KA_B = conditional_covariance(model, A, B, Kf, jitter=jitter)
    return 0.5 * (logdet_hpd(KAA, jitter) - logdet_hpd(KA_B, jitter))


def conditional_mutual_information(model: GaussianDAG, A: Sequence[int], B: Sequence[int],
                                   C: Sequence[int], Kf: torch.Tensor | None = None,
                                   *, jitter: float = 0.0) -> torch.Tensor:
    """``I(V_A; V_B | V_C) = 1/2 [logdet K_{A|C} - logdet K_{A|BC}]`` (nats)."""
    Kf = _resolve(model, Kf)
    KA_C = conditional_covariance(model, A, C, Kf, jitter=jitter)
    KA_BC = conditional_covariance(model, A, list(B) + list(C), Kf, jitter=jitter)
    return 0.5 * (logdet_hpd(KA_C, jitter) - logdet_hpd(KA_BC, jitter))


def sample(model: GaussianDAG, N: int, generator: torch.Generator) -> torch.Tensor:
    """Draw ``N`` i.i.d. samples of ``V_all`` by topological propagation.

    Returns a ``(N, D)`` tensor of the model's dtype. Real and complex models are
    both supported; complex uses a circularly-symmetric innovation. Nodes with a
    positive-definite noise use the Cholesky factor (bit-identical to previous
    releases for a given generator seed); singular-noise nodes (deterministic
    mechanisms, ``validate="psd"`` models) fall back to an eigendecomposition
    factor, so they propagate their parents' values plus a degenerate innovation.
    """
    X = torch.zeros((N, model.D), dtype=model.dtype, device=model.device)
    chol = {j: psd_factor(model.noise[j]) for j in range(model.M)}
    for j in range(model.M):
        if model.dtype.is_complex:
            real_dtype = torch.float64 if model.dtype == torch.complex128 else torch.float32
            zr = torch.randn((N, model.dims[j]), dtype=real_dtype, generator=generator)
            zi = torch.randn((N, model.dims[j]), dtype=real_dtype, generator=generator)
            z = torch.complex(zr, zi) / (2.0 ** 0.5)
        else:
            z = torch.randn((N, model.dims[j]), dtype=model.dtype, generator=generator)
        v = z @ chol[j].mH
        for i in model.parents[j]:
            v = v + X[:, model.slc(i)] @ model.edges[(i, j)].mH
        if model.has_mean:
            v = v + model.mean[j]                     # node offset c_j (broadcast)
        X[:, model.slc(j)] = v
    return X
