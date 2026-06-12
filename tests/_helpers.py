"""Shared seeded generators and ground-truth oracles for the durable test suite.

Not a pytest fixtures module (no conftest): these are plain functions, matching
the reference suite's convention of module-local seeded helpers, de-duplicated
into one place because this suite is large. Every generator takes ``dtype`` so
the same test body runs under ``torch.float64`` and ``torch.complex128``.

The oracle utilities (``closed_form_full_cov``, ``precision_conditional``,
``empirical_regression``, ``fd_jacobian``) are deliberately INDEPENDENT code
paths from the library, so a bug in the library is not silently reproduced by
the test that checks it.
"""

from __future__ import annotations

import torch

GLOBAL_SEED = 20260610
ALL_DTYPES = (torch.float64, torch.complex128)

# Tolerance constants (mirror the reference suite where they overlap).
ATOL_EXACT = 1e-10
ATOL_TIGHT = 1e-12
ATOL_INVARIANT = 1e-9
PSD_TOL = -1e-10
RTOL_FD = 1e-6
RTOL_MC = 0.02


# --------------------------------------------------------------------------
# Seeded random generators (dtype-generic)
# --------------------------------------------------------------------------
def gen(seed: int) -> torch.Generator:
    """A CPU torch.Generator seeded deterministically."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def randn_dt(*shape: int, seed: int, dtype: torch.dtype, scale: float = 1.0) -> torch.Tensor:
    """Random tensor of the given dtype. Complex uses independent re/im parts."""
    g = gen(seed)
    if dtype.is_complex:
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        re = torch.randn(shape, dtype=real_dtype, generator=g)
        im = torch.randn(shape, dtype=real_dtype, generator=g)
        return scale * torch.complex(re, im)
    return scale * torch.randn(shape, dtype=dtype, generator=g)


def spd(d: int, *, seed: int, dtype: torch.dtype, scale: float = 1.0, floor: float = 0.5) -> torch.Tensor:
    """Hermitian positive-definite ``scale*(L L^H)/d + floor*I``."""
    L = randn_dt(d, d, seed=seed, dtype=dtype)
    eye = torch.eye(d, dtype=dtype)
    return scale * (L @ L.mH) / d + floor * eye


def hermitian_psd(d: int, *, seed: int, dtype: torch.dtype) -> torch.Tensor:
    """Hermitian PD ``A A^H + I`` (parity with the reference helper)."""
    A = randn_dt(d, d, seed=seed, dtype=dtype)
    return A @ A.mH + torch.eye(d, dtype=dtype)


# --------------------------------------------------------------------------
# Topology builders -> (dims, edges, noise) in the GaussianDAG constructor format
#   edges: dict[(i, j)] = A_ji (parent i < child j)
#   noise: list, noise[j] = Sigma_j  (root cov for parentless nodes)
# --------------------------------------------------------------------------
def chain_dag(M: int, d: int, *, seed: int, dtype: torch.dtype):
    """Chain 0 -> 1 -> ... -> (M-1) with vector dimension ``d``."""
    dims = [d] * M
    edges = {(i, i + 1): randn_dt(d, d, seed=seed + i, dtype=dtype, scale=0.7)
             for i in range(M - 1)}
    noise = [spd(d, seed=seed + 100 + j, dtype=dtype, scale=0.5) for j in range(M)]
    return dims, edges, noise


def collider_dag(*, seed: int, dtype: torch.dtype, d: int = 1):
    """Collider 0 -> 2 <- 1 (two independent roots, common child)."""
    dims = [d, d, d]
    edges = {(0, 2): randn_dt(d, d, seed=seed, dtype=dtype, scale=1.0),
             (1, 2): randn_dt(d, d, seed=seed + 1, dtype=dtype, scale=0.7)}
    noise = [spd(d, seed=seed + 10, dtype=dtype),
             spd(d, seed=seed + 11, dtype=dtype),
             spd(d, seed=seed + 12, dtype=dtype, scale=0.3)]
    return dims, edges, noise


def diamond_dag(*, seed: int, dtype: torch.dtype, d: int = 2):
    """Diamond 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3."""
    dims = [d, d, d, d]
    edges = {
        (0, 1): randn_dt(d, d, seed=seed + 0, dtype=dtype, scale=0.6),
        (0, 2): randn_dt(d, d, seed=seed + 1, dtype=dtype, scale=0.6),
        (1, 3): randn_dt(d, d, seed=seed + 2, dtype=dtype, scale=0.6),
        (2, 3): randn_dt(d, d, seed=seed + 3, dtype=dtype, scale=0.6),
    }
    noise = [spd(d, seed=seed + 20 + j, dtype=dtype, scale=0.5) for j in range(4)]
    return dims, edges, noise


def diamond_tail_dag(*, seed: int, dtype: torch.dtype):
    """Diamond + tail with mixed vector dims [2,2,3,2,1] (the ST1 topology)."""
    dims = [2, 2, 3, 2, 1]
    edges = {
        (0, 1): randn_dt(dims[1], dims[0], seed=seed + 0, dtype=dtype),
        (0, 2): randn_dt(dims[2], dims[0], seed=seed + 1, dtype=dtype),
        (1, 3): randn_dt(dims[3], dims[1], seed=seed + 2, dtype=dtype),
        (2, 3): randn_dt(dims[3], dims[2], seed=seed + 3, dtype=dtype),
        (3, 4): randn_dt(dims[4], dims[3], seed=seed + 4, dtype=dtype),
    }
    noise = [spd(d, seed=seed + 30 + j, dtype=dtype) for j, d in enumerate(dims)]
    return dims, edges, noise


def multiroot_dag(*, seed: int, dtype: torch.dtype, d: int = 2):
    """Two roots {0, 1} both feeding 2, then 2 -> 3."""
    dims = [d, d, d, d]
    edges = {
        (0, 2): randn_dt(d, d, seed=seed + 0, dtype=dtype, scale=0.6),
        (1, 2): randn_dt(d, d, seed=seed + 1, dtype=dtype, scale=0.6),
        (2, 3): randn_dt(d, d, seed=seed + 2, dtype=dtype, scale=0.6),
    }
    noise = [spd(d, seed=seed + 40 + j, dtype=dtype, scale=0.5) for j in range(4)]
    return dims, edges, noise


def random_dag(M: int, *, seed: int, dtype: torch.dtype, density: float = 0.6,
               scale: float = 0.6, noise_scale: float = 0.5):
    """Random topological-order DAG with random scalar/vector node dims in {1,2,3}."""
    g = gen(seed)
    dims = [int(torch.randint(1, 4, (1,), generator=g)) for _ in range(M)]
    edges = {}
    ctr = 0
    for j in range(M):
        for i in range(j):
            if float(torch.rand(1, generator=g)) < density:
                edges[(i, j)] = randn_dt(dims[j], dims[i], seed=seed + 1000 + ctr,
                                         dtype=dtype, scale=scale)
                ctr += 1
        # ensure every non-first node has at least one parent unless it is a root
        if j > 0 and not any((i, j) in edges for i in range(j)):
            i = j - 1
            edges[(i, j)] = randn_dt(dims[j], dims[i], seed=seed + 2000 + j,
                                     dtype=dtype, scale=scale)
    noise = [spd(dims[j], seed=seed + 3000 + j, dtype=dtype, scale=noise_scale)
             for j in range(M)]
    return dims, edges, noise


# --------------------------------------------------------------------------
# Independent ground-truth oracles
# --------------------------------------------------------------------------
def closed_form_full_cov(dims, edges, noise, dtype: torch.dtype) -> torch.Tensor:
    """Full covariance via V_all = (I - Acal)^{-1} Z; Cov = T Sigma T^H.

    Independent re-derivation of the K-recursion (uses an explicit inverse).
    """
    offset = [0]
    for d in dims:
        offset.append(offset[-1] + d)
    D = offset[-1]
    Acal = torch.zeros((D, D), dtype=dtype)
    Sig = torch.zeros((D, D), dtype=dtype)
    M = len(dims)
    parents = {j: [i for (i, jj) in edges if jj == j] for j in range(M)}
    for j in range(M):
        Sig[offset[j]:offset[j + 1], offset[j]:offset[j + 1]] = noise[j]
        for i in parents[j]:
            Acal[offset[j]:offset[j + 1], offset[i]:offset[i + 1]] = edges[(i, j)]
    T = torch.linalg.inv(torch.eye(D, dtype=dtype) - Acal)
    return T @ Sig @ T.mH


def precision_conditional(Kfull: torch.Tensor, idx_A: torch.Tensor, idx_B: torch.Tensor):
    """Information-form conditioning: returns (mean_map, cond_cov).

    For the joint over (A, B): P = inv(K[idx][:,idx]); C = inv(P_AA);
    mean_map = -inv(P_AA) @ P_AB. Independent of the Schur-complement path.
    """
    idx = torch.cat([idx_A, idx_B])
    Ksub = Kfull[idx][:, idx]
    P = torch.linalg.inv(Ksub)
    da = idx_A.numel()
    PAA = P[:da, :da]
    PAB = P[:da, da:]
    PAA_inv = torch.linalg.inv(PAA)
    return -PAA_inv @ PAB, PAA_inv


def empirical_regression(VA: torch.Tensor, VB: torch.Tensor):
    """Least-squares mean map and residual covariance from samples."""
    N = VA.shape[0]
    M = (VA.mH @ VB) @ torch.linalg.inv(VB.mH @ VB)
    resid = VA - VB @ M.mH
    return M, (resid.mH @ resid) / N


def fd_jacobian(fn, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Central finite-difference Jacobian of a tensor-valued ``fn`` at ``x`` (real x).

    Returns shape ``(numel(fn(x)), numel(x))``.
    """
    x = x.detach()
    f0 = fn(x).reshape(-1)
    q = x.numel()
    cols = []
    for a in range(q):
        e = torch.zeros_like(x).reshape(-1)
        e[a] = eps
        e = e.reshape(x.shape)
        fa = (fn(x + e).reshape(-1) - fn(x - e).reshape(-1)) / (2 * eps)
        cols.append(fa)
    return torch.stack(cols, dim=1)


# --------------------------------------------------------------------------
# Predicates / metrics
# --------------------------------------------------------------------------
def is_hermitian(A: torch.Tensor, atol: float = 1e-10) -> bool:
    return bool(torch.allclose(A, A.mH, atol=atol))


def is_psd(A: torch.Tensor, tol: float = PSD_TOL) -> bool:
    w = torch.linalg.eigvalsh(0.5 * (A + A.mH))
    return bool(w.min().real > tol)


def relerr_fro(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = torch.linalg.norm(expected)
    return float(torch.linalg.norm(actual - expected) / torch.clamp(denom, min=1e-12))
