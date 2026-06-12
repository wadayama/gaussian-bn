"""Tests for gaussian_bn.optimize: PGA and natural-gradient ascent."""

from __future__ import annotations

import pytest
import torch

from gaussian_bn.optimize import (
    natural_gradient_ascent,
    natural_gradient_step,
    pga_ascent,
)
from gaussian_bn.projections import project_frobenius_ball


def test_pga_increases_objective():
    # maximize f(x) = -||x - c||^2 ; ascent must increase f
    c = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    def obj():
        return -((x - c) ** 2).sum()

    hist = pga_ascent(obj, [x], step_size=0.1, num_iters=50)
    assert hist[-1] > hist[0]
    assert torch.allclose(x.detach(), c, atol=1e-3)


def test_pga_respects_projector():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float64, requires_grad=True)  # norm 5

    def obj():
        return (x ** 2).sum()  # wants to grow; projector caps it

    def proj(params):
        params[0].copy_(project_frobenius_ball(params[0], P=1.0))

    pga_ascent(obj, [x], step_size=0.1, num_iters=20, projector=proj)
    assert float(torch.linalg.norm(x.detach()) ** 2) <= 1.0 + 1e-8


def test_pga_rejects_invalid_args():
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        pga_ascent(lambda: (x ** 2).sum(), [x], step_size=0.0, num_iters=5)
    with pytest.raises(ValueError):
        pga_ascent(lambda: (x ** 2).sum(), [x], step_size=0.1, num_iters=0)
    y = torch.zeros(2)  # no grad
    with pytest.raises(ValueError):
        pga_ascent(lambda: (y ** 2).sum(), [y], step_size=0.1, num_iters=5)


def test_natural_gradient_step_reduces_to_gradient_when_metric_identity():
    eta = torch.tensor([1.0, 2.0], dtype=torch.float64)
    grad = torch.tensor([0.3, -0.7], dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    out = natural_gradient_step(eta, grad, I, step_size=0.5)
    assert torch.allclose(out, eta + 0.5 * grad, atol=1e-12)


def test_natural_gradient_one_step_exact_for_quadratic():
    # f(eta) = -(eta - t)^T M (eta - t); grad = -2 M (eta - t); metric = M
    # natural step with step_size 0.5: eta + 0.5 * M^{-1}(-2 M (eta - t)) = t
    t = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    M = torch.tensor([[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]], dtype=torch.float64)

    def obj(eta):
        d = eta - t
        return -(d @ (M @ d))

    def metric(eta):
        return M

    eta_final, hist = natural_gradient_ascent(obj, metric, torch.zeros(3, dtype=torch.float64),
                                              step_size=0.5, num_iters=1)
    assert torch.allclose(eta_final, t, atol=1e-10)


def test_natural_gradient_ascent_increases_objective():
    t = torch.tensor([1.0, -1.0], dtype=torch.float64)
    M = torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64)

    def obj(eta):
        d = eta - t
        return -(d @ (M @ d))

    eta0 = torch.tensor([-2.0, 2.0], dtype=torch.float64)
    _, hist = natural_gradient_ascent(obj, lambda e: M, eta0, step_size=0.1, num_iters=10)
    assert hist[-1] > hist[0]
