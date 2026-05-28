"""Minimal FlashLLA usage: causal Local Linear Attention forward + backward.

Run on a single CUDA GPU:
    python examples/minimal_usage.py
"""

import torch

from flashlla import LLAFunction, lla_attention


def main() -> None:
    assert torch.cuda.is_available(), "FlashLLA kernels require a CUDA device."

    torch.manual_seed(0)
    B, S, D = 2, 256, 64
    q = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    # Functional API (most concise path).
    o = lla_attention(q, k, v, ridge_lambda=0.1)
    print(f"[functional] output: shape={tuple(o.shape)} dtype={o.dtype}")

    # Backward through an arbitrary scalar loss.
    loss = o.float().pow(2).mean()
    loss.backward()
    print(
        f"[functional] grad norms: "
        f"|gq|={q.grad.norm().item():.4f}, "
        f"|gk|={k.grad.norm().item():.4f}, "
        f"|gv|={v.grad.norm().item():.4f}"
    )

    # Equivalent explicit autograd.Function call, exposing the omega normalizer.
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ridge = torch.tensor(0.1, device="cuda", dtype=torch.bfloat16)
    o2, omega = LLAFunction.apply(
        q2, k2, v2, ridge,
        D ** -0.5,   # qk_scale
        1e-6,        # delta_eps
        1e-6, 1e-6,  # cg_atol, cg_rtol
        32,          # cg_max_iters
        False,       # cg_use_preconditioner
    )
    print(
        f"[LLAFunction] output: shape={tuple(o2.shape)}, "
        f"omega mean={omega.float().mean().item():.4e}"
    )


if __name__ == "__main__":
    main()
