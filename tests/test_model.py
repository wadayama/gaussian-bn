"""Tests for gaussian_bn.model: GaussianDAG container, k_full, pack/unpack."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.model import (
    GaussianDAG,
    full_covariance_closed_form,
    k_full,
    pack,
    unpack,
)
from gaussian_bn.linalg import hermitianize

from _helpers import (
    ALL_DTYPES,
    ATOL_EXACT,
    ATOL_TIGHT,
    diamond_dag,
    diamond_tail_dag,
    is_hermitian,
    multiroot_dag,
    relerr_fro,
    spd,
)


def _model(builder, *, seed, dtype, **kw):
    dims, edges, noise = builder(seed=seed, dtype=dtype, **kw)
    return GaussianDAG(dims, edges, noise, dtype=dtype)


# --------------------------------------------------------------------------
# structure / indexing
# --------------------------------------------------------------------------
def test_offsets_and_D():
    m = _model(diamond_tail_dag, seed=1, dtype=torch.float64)
    assert m.offset == [0, 2, 4, 7, 9, 10]
    assert m.D == 10


def test_node_index_and_slc():
    m = _model(diamond_tail_dag, seed=1, dtype=torch.float64)
    idx = m.node_index([1, 3])
    assert idx.tolist() == [2, 3, 7, 8]
    assert m.slc(2) == slice(4, 7)


def test_parents_sorted_and_roots():
    m = _model(multiroot_dag, seed=2, dtype=torch.float64)
    assert m.roots == [0, 1]
    assert m.parents[2] == [0, 1]
    assert m.parents[3] == [2]


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_k_full_vs_closed_form(dtype):
    m = _model(diamond_tail_dag, seed=11, dtype=dtype)
    Kf = k_full(m)
    ref = full_covariance_closed_form(m)
    assert relerr_fro(Kf, ref) < 1e-10
    assert is_hermitian(Kf, atol=ATOL_EXACT)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_dtype_preserved(dtype):
    m = _model(diamond_dag, seed=3, dtype=dtype)
    assert k_full(m).dtype == dtype


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def test_reject_edge_wrong_topo_order():
    with pytest.raises(ValueError):
        GaussianDAG([1, 1], {(1, 0): torch.eye(1)}, [torch.eye(1), torch.eye(1)])


def test_reject_edge_shape_mismatch():
    # edge (0,1) should be dims[1] x dims[0] = 2 x 1
    with pytest.raises(ValueError):
        GaussianDAG([1, 2], {(0, 1): torch.eye(3)},
                    [torch.eye(1), torch.eye(2)])


def test_reject_non_spd_noise():
    bad = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError):
        GaussianDAG([2], {}, [bad])


def test_reject_nonsquare_noise():
    with pytest.raises(ValueError):
        GaussianDAG([2], {}, [torch.zeros(2, 3)])


def test_reject_wrong_noise_length():
    with pytest.raises(ValueError):
        GaussianDAG([1, 1], {(0, 1): torch.eye(1)}, [torch.eye(1)])


# --------------------------------------------------------------------------
# pack / unpack round-trips
# --------------------------------------------------------------------------
@pytest.mark.parametrize("noise_param", ["chol", "logdiag", "fixed"])
def test_pack_unpack_roundtrip(noise_param):
    # use diagonal noise so logdiag can represent it exactly
    dims = [2, 2, 2]
    edges = {(0, 1): torch.randn(2, 2, dtype=torch.float64),
             (1, 2): torch.randn(2, 2, dtype=torch.float64)}
    noise = [torch.diag(torch.tensor([1.3, 0.7])),
             torch.diag(torch.tensor([0.9, 1.1])),
             torch.diag(torch.tensor([0.5, 2.0]))]
    m = GaussianDAG(dims, edges, noise)
    eta, pk = pack(m, noise_param=noise_param)
    m2 = unpack(eta, pk, m)
    for key in edges:
        assert torch.allclose(m2.edges[key], m.edges[key], atol=ATOL_TIGHT)
    if noise_param != "fixed":
        for j in range(m.M):
            assert torch.allclose(m2.noise[j], m.noise[j], atol=1e-8)


def test_pack_unpack_chol_full_noise_roundtrip():
    # full (non-diagonal) SPD noise round-trips exactly under chol
    dims = [2, 3]
    edges = {(0, 1): torch.randn(3, 2, dtype=torch.float64)}
    noise = [spd(2, seed=1, dtype=torch.float64),
             spd(3, seed=2, dtype=torch.float64)]
    m = GaussianDAG(dims, edges, noise)
    eta, pk = pack(m, noise_param="chol")
    m2 = unpack(eta, pk, m)
    for j in range(m.M):
        assert torch.allclose(hermitianize(m2.noise[j]), hermitianize(m.noise[j]), atol=1e-8)


def test_pack_noise_is_spd_for_any_eta():
    # perturb eta arbitrarily; chol parametrization must stay SPD
    dims = [2, 2]
    edges = {(0, 1): torch.randn(2, 2, dtype=torch.float64)}
    noise = [spd(2, seed=1, dtype=torch.float64), spd(2, seed=2, dtype=torch.float64)]
    m = GaussianDAG(dims, edges, noise)
    eta, pk = pack(m, noise_param="chol")
    eta2 = eta.detach() + 3.0 * torch.randn_like(eta)
    m2 = unpack(eta2, pk, m)
    for j in range(m.M):
        w = torch.linalg.eigvalsh(hermitianize(m2.noise[j]))
        assert w.min() > 0


def test_pack_free_edges_subset():
    m = _model(diamond_dag, seed=5, dtype=torch.float64)
    free = [(0, 1), (1, 3)]
    eta, pk = pack(m, free_edges=free, noise_param="fixed")
    # only the two free edges are represented
    assert pk.free_edges == free
    # unpack with modified eta changes only those edges
    eta2 = eta.detach().clone()
    eta2[:] += 0.1
    m2 = unpack(eta2, pk, m)
    assert not torch.allclose(m2.edges[(0, 1)], m.edges[(0, 1)])
    assert torch.allclose(m2.edges[(0, 2)], m.edges[(0, 2)], atol=ATOL_TIGHT)


def test_pack_unpack_differentiable():
    m = _model(diamond_dag, seed=6, dtype=torch.float64)
    eta, pk = pack(m, noise_param="chol")
    m2 = unpack(eta, pk, m)
    loss = k_full(m2).pow(2).sum()
    loss.backward()
    assert eta.grad is not None
    assert eta.grad.abs().sum() > 1e-6
