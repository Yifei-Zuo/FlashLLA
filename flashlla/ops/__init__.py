from __future__ import annotations

import torch

from flashlla.ops.backward import lla_backward
from flashlla.ops.forward import lla_forward

__all__ = ["LLAFunction", "lla_attention"]


class LLAFunction(torch.autograd.Function):
    """Autograd wrapper around the Triton LLA kernels; returns ``(o, omega)``."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ridge_lambda: torch.Tensor,
        qk_scale: float,
        delta_eps: float,
        cg_atol: float,
        cg_rtol: float,
        cg_max_iters: int,
        cg_use_preconditioner: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, r, d, m, w = lla_forward(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            ridge_lambda,
            qk_scale,
            delta_eps,
            cg_atol,
            cg_rtol,
            cg_max_iters,
            cg_use_preconditioner,
        )
        qk_scale_tensor = torch.tensor(qk_scale, device=q.device, dtype=q.dtype)
        cg_atol_tensor = torch.tensor(cg_atol, device=q.device, dtype=q.dtype)
        cg_rtol_tensor = torch.tensor(cg_rtol, device=q.device, dtype=q.dtype)
        cg_max_iters_tensor = torch.tensor(cg_max_iters, device=q.device, dtype=q.dtype)
        cg_use_preconditioner_tensor = torch.tensor(cg_use_preconditioner, device=q.device, dtype=torch.bool)
        ctx.save_for_backward(
            q, k, v, r, d, m,
            ridge_lambda,
            qk_scale_tensor,
            cg_atol_tensor,
            cg_rtol_tensor,
            cg_max_iters_tensor,
            cg_use_preconditioner_tensor,
        )
        return o, w

    @staticmethod
    def backward(ctx, grad_o, grad_w):
        q, k, v, r, d, m, ridge_lambda, qk_scale, cg_atol, cg_rtol, cg_max_iters, cg_use_preconditioner = ctx.saved_tensors
        grad_q, grad_k, grad_v = lla_backward(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            r, d, m, grad_o,
            ridge_lambda,
            qk_scale.item(),
            cg_atol.item(),
            cg_rtol.item(),
            int(cg_max_iters.item()),
            bool(cg_use_preconditioner.item()),
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None


def lla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    ridge_lambda: float | torch.Tensor = 0.01,
    qk_scale: float | None = None,
    delta_eps: float = 1e-12,
    cg_atol: float = 1e-12,
    cg_rtol: float = 1e-12,
    cg_max_iters: int = 32,
    cg_use_preconditioner: bool = False,
) -> torch.Tensor:
    """Functional wrapper around :class:`LLAFunction`.

    Args:
        q, k, v: ``(batch, seqlen, head_dim)`` tensors on CUDA.
        ridge_lambda: scalar or ``(1,)`` / ``(batch, seqlen)`` tensor. Coerced
            to a tensor matching ``q``'s device/dtype.
        qk_scale: defaults to ``1 / sqrt(head_dim)`` when ``None``.

    Returns:
        Attention output ``o`` with shape ``(batch, seqlen, head_dim)``.
    """
    if qk_scale is None:
        qk_scale = q.shape[-1] ** -0.5
    if not isinstance(ridge_lambda, torch.Tensor):
        ridge_lambda = torch.tensor(ridge_lambda, device=q.device, dtype=q.dtype)
    o, _ = LLAFunction.apply(
        q, k, v,
        ridge_lambda,
        float(qk_scale),
        delta_eps,
        cg_atol,
        cg_rtol,
        cg_max_iters,
        cg_use_preconditioner,
    )
    return o
