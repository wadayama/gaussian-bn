"""Tests for gaussian_bn.projections: Frobenius-ball and total-power projectors."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.projections import project_frobenius_ball, project_total_power

from _helpers import ALL_DTYPES


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_frobenius_ball_inside_is_identity(dtype):
    A = 0.1 * torch.randn(3, 2, dtype=torch.float64).to(dtype)
    out = project_frobenius_ball(A, P=100.0)
    assert torch.allclose(out, A, atol=1e-14)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_frobenius_ball_outside_on_boundary(dtype):
    A = 5.0 * torch.ones(2, 2, dtype=torch.float64).to(dtype)
    P = 4.0
    out = project_frobenius_ball(A, P=P)
    assert abs(float(torch.linalg.norm(out) ** 2) - P) < 1e-10
    # direction preserved
    assert torch.allclose(out, A * (P ** 0.5 / float(torch.linalg.norm(A))), atol=1e-10)


def test_frobenius_ball_rejects_nonpositive_P():
    with pytest.raises(ValueError):
        project_frobenius_ball(torch.eye(2), P=0.0)


def test_total_power_outside_uniform_rescale():
    params = [torch.ones(2, 2, dtype=torch.float64), 2.0 * torch.ones(1, 2, dtype=torch.float64)]
    P = 4.0
    out = project_total_power(params, P=P)
    total = sum(float(torch.linalg.norm(p) ** 2) for p in out)
    assert abs(total - P) < 1e-10


def test_total_power_inside_is_identity():
    params = [0.1 * torch.ones(2, 2), 0.1 * torch.ones(2, 2)]
    out = project_total_power(params, P=100.0)
    for a, b in zip(out, params):
        assert torch.allclose(a, b, atol=1e-14)


def test_projection_idempotent():
    A = 5.0 * torch.ones(2, 2, dtype=torch.float64)
    once = project_frobenius_ball(A, P=4.0)
    twice = project_frobenius_ball(once, P=4.0)
    assert torch.allclose(once, twice, atol=1e-12)


def test_total_power_rejects_empty_and_nonpositive():
    with pytest.raises(ValueError):
        project_total_power([], P=1.0)
    with pytest.raises(ValueError):
        project_total_power([torch.eye(2)], P=-1.0)
