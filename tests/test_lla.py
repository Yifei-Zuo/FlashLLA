"""Correctness tests for the original LLA kernel.

Compares the Triton ``LLAFunction`` against the pure-PyTorch reference in
:mod:`flashlla.ops.naive` on tiny CUDA tensors. The kernel runs in bfloat16
internally; the naive reference runs in fp32 with ``torch.linalg.solve``,
so tolerances are loose to absorb the precision gap.
"""

from __future__ import annotations

import pytest
import torch

from flashlla import LLAFunction
from flashlla.ops.naive import lla_backward_naive, lla_forward_naive


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashLLA kernels require a CUDA device.",
)


SHAPES = [
    (1, 64, 64),
    (1, 128, 64),
    (1, 256, 64),
    (1, 512, 64),
    (1, 1024, 64),
]

# Ridge is intentionally large enough to keep the CG system well-conditioned
# in bfloat16; the absolute value isn't load-bearing for correctness.
RIDGE_LAMBDA = 10.0
DELTA_EPS = 1e-6
CG_TOL = 1e-6
CG_MAX_ITERS = 32


def _make_qkv(B: int, S: int, D: int, *, seed: int, requires_grad: bool = False):
    torch.manual_seed(seed)
    q = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=requires_grad)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=requires_grad)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.float32, requires_grad=requires_grad)
    return q, k, v


def _kernel_args(q, k, v, qk_scale):
    ridge = torch.tensor(RIDGE_LAMBDA, device=q.device, dtype=q.dtype)
    return (q, k, v, ridge, qk_scale, DELTA_EPS, CG_TOL, CG_TOL, CG_MAX_ITERS, False)


@pytest.mark.parametrize("B,S,D", SHAPES)
def test_forward_matches_naive(B: int, S: int, D: int) -> None:
    """Triton forward output should match the fp32 naive reference."""
    q, k, v = _make_qkv(B, S, D, seed=0)
    qk_scale = D ** -0.5

    o_kernel, _ = LLAFunction.apply(*_kernel_args(q, k, v, qk_scale))
    o_naive, _, _, _ = lla_forward_naive(
        q, k, v, qk_scale, RIDGE_LAMBDA, DELTA_EPS,
    )

    # Loose tolerances: kernel runs in bfloat16 + CG, naive solves in fp32.
    # A small fraction of near-zero outputs disagree by O(0.1) absolutely.
    torch.testing.assert_close(
        o_kernel.float(), o_naive,
        rtol=1e-2, atol=1e-2,
    )


@pytest.mark.parametrize("B,S,D", SHAPES)
def test_backward_matches_naive(B: int, S: int, D: int) -> None:
    """Autograd-computed grads should match the fp32 naive backward."""
    q, k, v = _make_qkv(B, S, D, seed=0, requires_grad=True)
    qk_scale = D ** -0.5

    o_kernel, _ = LLAFunction.apply(*_kernel_args(q, k, v, qk_scale))
    torch.manual_seed(42)
    grad_o = torch.randn_like(o_kernel)
    o_kernel.backward(grad_o)
    gq_kernel = q.grad.detach().float()
    gk_kernel = k.grad.detach().float()
    gv_kernel = v.grad.detach().float()

    with torch.no_grad():
        _, r_ref, d_ref, m_ref = lla_forward_naive(
            q.detach(), k.detach(), v.detach(),
            qk_scale, RIDGE_LAMBDA, DELTA_EPS,
        )
        gq_naive, gk_naive, gv_naive = lla_backward_naive(
            q.detach(), k.detach(), v.detach(),
            r_ref, d_ref, m_ref, grad_o.float(),
            qk_scale, RIDGE_LAMBDA,
        )

    torch.testing.assert_close(gq_kernel, gq_naive, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(gk_kernel, gk_naive, rtol=1e-2, atol=1e-1)
    torch.testing.assert_close(gv_kernel, gv_naive, rtol=1e-2, atol=1e-1)
