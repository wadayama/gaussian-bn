"""Tests for gaussian_bn.krecursion: compute_k_blocks, get_K, and Hermitian flip.

These exercise the recursion directly (independent of the GaussianDAG model) by
building the (parents, edge_mats, root_covs, noise_covs) inputs from the shared
topology helpers, and cross-check the assembled full covariance against the
independent (I - Acal)^{-1} oracle.
"""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.krecursion import compute_k_blocks, get_K
from gaussian_bn.linalg import hermitianize

from _helpers import (
    ALL_DTYPES,
    ATOL_EXACT,
    ATOL_TIGHT,
    chain_dag,
    closed_form_full_cov,
    collider_dag,
    diamond_dag,
    diamond_tail_dag,
    is_hermitian,
    multiroot_dag,
    random_dag,
    relerr_fro,
)


# --------------------------------------------------------------------------
# helpers: convert (dims, edges, noise) -> recursion inputs and assemble full K
# --------------------------------------------------------------------------
def _inputs(dims, edges, noise):
    M = len(dims)
    parents = {j: sorted(i for (i, jj) in edges if jj == j) for j in range(M)}
    roots = [j for j in range(M) if not parents[j]]
    edge_mats = {(j, i): edges[(i, j)] for (i, j) in edges}
    root_covs = {r: noise[r] for r in roots}
    noise_covs = {j: noise[j] for j in range(M) if j not in roots}
    return M, parents, edge_mats, root_covs, noise_covs


def _assemble_full(dims, K):
    M = len(dims)
    offset = [0]
    for d in dims:
        offset.append(offset[-1] + d)
    D = offset[-1]
    Kf = torch.zeros((D, D), dtype=K[(0, 0)].dtype)
    for j in range(M):
        for k in range(M):
            Kf[offset[j]:offset[j + 1], offset[k]:offset[k + 1]] = get_K(K, j, k)
    return Kf


def _blocks(dims, edges, noise):
    M, parents, edge_mats, root_covs, noise_covs = _inputs(dims, edges, noise)
    return compute_k_blocks(M, parents, edge_mats, root_covs, noise_covs)


# --------------------------------------------------------------------------
# key coverage / structure
# --------------------------------------------------------------------------
def test_key_coverage_diamond():
    dims, edges, noise = diamond_dag(seed=1, dtype=torch.float64)
    K = _blocks(dims, edges, noise)
    expected = {(j, k) for j in range(len(dims)) for k in range(j + 1)}
    assert set(K.keys()) == expected


def test_key_coverage_multiroot():
    dims, edges, noise = multiroot_dag(seed=2, dtype=torch.float64)
    K = _blocks(dims, edges, noise)
    expected = {(j, k) for j in range(len(dims)) for k in range(j + 1)}
    assert set(K.keys()) == expected
    # the two roots {0,1} are independent: cross block is exactly zero
    assert torch.allclose(get_K(K, 1, 0), torch.zeros_like(get_K(K, 1, 0)), atol=ATOL_TIGHT)
    # each root self-block equals its own (hermitianized) covariance
    assert torch.allclose(K[(0, 0)], hermitianize(noise[0]), atol=ATOL_EXACT)
    assert torch.allclose(K[(1, 1)], hermitianize(noise[1]), atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_get_K_hermitian_flip(dtype):
    dims, edges, noise = diamond_dag(seed=3, dtype=dtype)
    K = _blocks(dims, edges, noise)
    assert torch.equal(get_K(K, 1, 3), K[(3, 1)].mH)
    assert torch.equal(get_K(K, 3, 1), K[(3, 1)])


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_hermitianize_idempotent_and_hermitian(dtype):
    A = torch.randn(4, 4, dtype=torch.float64)
    if dtype.is_complex:
        A = torch.complex(A, torch.randn(4, 4, dtype=torch.float64))
    H = hermitianize(A)
    assert is_hermitian(H, atol=ATOL_TIGHT)
    assert torch.allclose(hermitianize(H), H, atol=ATOL_TIGHT)


# --------------------------------------------------------------------------
# deterministic closed-form recomputations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_chain_deterministic(dtype):
    # chain 0 -> 1 -> 2, hand closed form for the self/cross blocks
    dims, edges, noise = chain_dag(3, 2, seed=5, dtype=dtype)
    K = _blocks(dims, edges, noise)
    A10, A21 = edges[(0, 1)], edges[(1, 2)]
    Sx, S1, S2 = noise[0], noise[1], noise[2]
    K00 = hermitianize(Sx)
    K11 = hermitianize(A10 @ K00 @ A10.mH + S1)
    K22 = hermitianize(A21 @ K11 @ A21.mH + S2)
    K10 = A10 @ K00
    K20 = A21 @ K10
    K21 = A21 @ K11
    assert torch.allclose(K[(0, 0)], K00, atol=ATOL_EXACT)
    assert torch.allclose(K[(1, 1)], K11, atol=ATOL_EXACT)
    assert torch.allclose(K[(2, 2)], K22, atol=ATOL_EXACT)
    assert torch.allclose(K[(1, 0)], K10, atol=ATOL_EXACT)
    assert torch.allclose(K[(2, 0)], K20, atol=ATOL_EXACT)
    assert torch.allclose(K[(2, 1)], K21, atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_diamond_cross_covariance_nonzero(dtype):
    # K_{21}: nodes 1 and 2 share ancestor 0, so cross-cov must be non-zero
    dims, edges, noise = diamond_dag(seed=7, dtype=dtype)
    K = _blocks(dims, edges, noise)
    A10, A20 = edges[(0, 1)], edges[(0, 2)]
    K21_expected = A20 @ hermitianize(noise[0]) @ A10.mH
    assert torch.allclose(get_K(K, 2, 1), K21_expected, atol=ATOL_EXACT)
    assert torch.linalg.norm(get_K(K, 2, 1)) > 1e-6


# --------------------------------------------------------------------------
# full assembly vs independent (I - Acal)^{-1} oracle
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full_cov_vs_closed_form_diamond_tail(dtype):
    dims, edges, noise = diamond_tail_dag(seed=11, dtype=dtype)
    Kf = _assemble_full(dims, _blocks(dims, edges, noise))
    ref = closed_form_full_cov(dims, edges, noise, dtype)
    assert relerr_fro(Kf, ref) < 1e-10
    assert is_hermitian(Kf, atol=ATOL_EXACT)
    assert torch.linalg.eigvalsh(hermitianize(Kf)).min().real > 0


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_full_cov_vs_closed_form_random(dtype, seed):
    dims, edges, noise = random_dag(6, seed=seed, dtype=dtype)
    Kf = _assemble_full(dims, _blocks(dims, edges, noise))
    ref = closed_form_full_cov(dims, edges, noise, dtype)
    assert relerr_fro(Kf, ref) < 1e-9


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_full_cov_vs_closed_form_multiroot(dtype):
    dims, edges, noise = multiroot_dag(seed=13, dtype=dtype)
    Kf = _assemble_full(dims, _blocks(dims, edges, noise))
    ref = closed_form_full_cov(dims, edges, noise, dtype)
    assert relerr_fro(Kf, ref) < 1e-10


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_self_blocks_hermitian(dtype):
    dims, edges, noise = random_dag(7, seed=21, dtype=dtype)
    K = _blocks(dims, edges, noise)
    for j in range(len(dims)):
        assert is_hermitian(K[(j, j)], atol=ATOL_EXACT)


def test_single_node_trivial():
    dims = [3]
    K = compute_k_blocks(1, {0: []}, {}, {0: torch.eye(3, dtype=torch.float64) * 2.0}, {})
    assert torch.allclose(K[(0, 0)], 2.0 * torch.eye(3, dtype=torch.float64), atol=ATOL_TIGHT)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_rejects_nontopological_parent():
    with pytest.raises(ValueError):
        compute_k_blocks(
            3,
            {2: [3]},  # parent index >= child
            {(2, 3): torch.eye(1)},
            {0: torch.eye(1), 1: torch.eye(1)},
            {2: torch.eye(1)},
        )


def test_rejects_no_root():
    # every node claims a parent -> no root
    with pytest.raises(ValueError):
        compute_k_blocks(
            2,
            {0: [1], 1: [0]},
            {(0, 1): torch.eye(1), (1, 0): torch.eye(1)},
            {},
            {0: torch.eye(1), 1: torch.eye(1)},
        )


def test_rejects_mismatched_root_keys():
    dims, edges, noise = diamond_dag(seed=1, dtype=torch.float64)
    M, parents, edge_mats, root_covs, noise_covs = _inputs(dims, edges, noise)
    bad_roots = dict(root_covs)
    bad_roots[1] = torch.eye(dims[1], dtype=torch.float64)  # node 1 is not a root
    with pytest.raises(ValueError):
        compute_k_blocks(M, parents, edge_mats, bad_roots, noise_covs)
