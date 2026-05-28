from __future__ import annotations

from typing import Tuple

import torch


def _relmm(X: torch.Tensor, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    XK = X @ K.transpose(-1, -2)
    XQ = (X * Q).sum(dim=-1, keepdim=True)
    return XK - XQ


def lla_forward_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float,
    ridge_lambda: float | torch.Tensor,
    delta_eps: float,
    lla_block_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference forward pass for causal Local Linear Attention.

    Returns ``(o, rho, denom, m)`` matching the intermediates saved by
    ``lla_forward`` for use by the backward.
    """
    sq, sk = q.shape[-2], k.shape[-2]
    batch_size, _, dim = q.shape
    if lla_block_size is None:
        lla_block_size = dim
    assert dim % lla_block_size == 0 and lla_block_size <= dim, (
        f"dim must be divisible by lla_block_size; got dim={dim}, "
        f"lla_block_size={lla_block_size}"
    )
    assert q.dtype == k.dtype == v.dtype, (
        f"dtypes must match; got q={q.dtype}, k={k.dtype}, v={v.dtype}"
    )

    row_offset = torch.arange(sq, device=q.device).view(-1, 1)
    col_offset = torch.arange(sk, device=k.device).view(1, -1)
    attention_mask = row_offset >= col_offset
    qk = (q @ k.transpose(-1, -2)) * qk_scale
    qk = qk.masked_fill(~attention_mask, float("-inf"))
    m = qk.max(dim=-1, keepdim=True).values
    weight = torch.exp(qk - m)
    omega = weight.sum(dim=-1, keepdim=True)
    tilde_mu = torch.einsum("bij,bjd->bid", weight, k)
    mu = tilde_mu - omega * q
    q_blk = q.contiguous().view(batch_size, sq, dim // lla_block_size, lla_block_size)
    k_blk = k.contiguous().view(batch_size, sk, dim // lla_block_size, lla_block_size)
    mu_blk = mu.contiguous().view(batch_size, sq, dim // lla_block_size, lla_block_size)
    tilde_mu_blk = tilde_mu.contiguous().view(batch_size, sq, dim // lla_block_size, lla_block_size)
    tilde_sigma = torch.einsum("bij,bjtd,bjte->bitde", weight, k_blk, k_blk)
    A = torch.einsum("bitd,bite->bitde", tilde_mu_blk, q_blk)
    B = torch.einsum("bitd,bite->bitde", q_blk, tilde_mu_blk)
    C = omega.unsqueeze(-1).unsqueeze(-1) * torch.einsum("bitd,bite->bitde", q_blk, q_blk)
    sigma = tilde_sigma - A - B + C
    eye = torch.eye(lla_block_size, device=q.device, dtype=q.dtype).view(
        1, 1, lla_block_size, lla_block_size
    )

    if isinstance(ridge_lambda, torch.Tensor) and ridge_lambda.ndim == 1:
        ridge_lambda = ridge_lambda.expand(batch_size, sq).unsqueeze(-1)

    sigma = sigma + eye * (ridge_lambda * omega).unsqueeze(-1).unsqueeze(-1)
    sigma = sigma.to(torch.float32)
    mu_blk = mu_blk.to(torch.float32)
    rho = torch.linalg.solve(sigma, mu_blk).to(q.dtype)
    rho = rho.contiguous().view(batch_size, sq, dim)
    mu = mu_blk.contiguous().view(batch_size, sq, dim)
    denom = omega - (mu * rho).sum(dim=-1, keepdim=True)
    denom = denom + delta_eps * torch.sign(denom)
    numer = 1 - torch.einsum("bid,bjd->bij", rho, k) + (rho * q).sum(dim=-1, keepdim=True)
    S = (weight * numer / denom).to(v.dtype)
    out = torch.einsum("bij,bjd->bid", S, v)
    return out, rho, denom, m


@torch.no_grad()
def lla_backward_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    m: torch.Tensor,
    grad_o: torch.Tensor,
    qk_scale: float,
    ridge_lambda: float | torch.Tensor,
    lla_block_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference backward pass; returns ``(grad_q, grad_k, grad_v)``."""
    batch_size, sq, dim = q.shape
    sk = k.shape[1]
    if lla_block_size is None:
        lla_block_size = dim
    assert dim == lla_block_size, (
        "lla_backward_naive currently requires lla_block_size == dim; "
        f"got dim={dim}, lla_block_size={lla_block_size}"
    )

    row_offset = torch.arange(sq, device=q.device).view(-1, 1)
    col_offset = torch.arange(sk, device=k.device).view(1, -1)
    attention_mask = row_offset >= col_offset
    qk = (q @ k.transpose(-1, -2)) * qk_scale
    qk = qk.masked_fill(~attention_mask, float("-inf"))
    W = torch.exp(qk - m)
    S = W * (1 - _relmm(r, q, k)) / d
    S = S.to(grad_o.dtype)
    G = torch.einsum("bid,bjd->bij", grad_o, v)
    C = G * W / d
    beta = (G * S).sum(dim=-1, keepdim=True) / d
    tau = C.sum(dim=-1, keepdim=True)
    T = torch.einsum("bij,bjd->bid", C, k) - tau * q
    omega = W.sum(dim=-1, keepdim=True)
    tilde_mu = torch.einsum("bij,bjd->bid", W, k)
    tilde_sigma = torch.einsum("bij,bjd,bje->bide", W, k, k)
    A = torch.einsum("bid,bie->bide", tilde_mu, q)
    B = torch.einsum("bid,bie->bide", q, tilde_mu)
    C_ = omega.unsqueeze(-1) * torch.einsum("bid,bie->bide", q, q)
    sigma = tilde_sigma - A - B + C_
    eye = torch.eye(lla_block_size, device=q.device, dtype=q.dtype).view(
        1, 1, lla_block_size, lla_block_size
    )
    if isinstance(ridge_lambda, torch.Tensor) and ridge_lambda.ndim == 1:
        ridge_lambda = ridge_lambda.expand(batch_size, sq).unsqueeze(-1)
    sigma = sigma + eye * (ridge_lambda * omega).unsqueeze(-1)
    U = torch.linalg.solve(sigma, T)
    DM = -U + 2 * beta * r
    relmm_R = _relmm(r, q, k)
    relmm_U = _relmm(U, q, k)
    relmm_DM = _relmm(DM, q, k)
    W_factor = W * (
        -beta + relmm_DM
        - 0.5 * relmm_DM * relmm_R
        + 0.5 * relmm_U * relmm_R
    )

    grad_q = torch.einsum("bij,bjd->bid", (G * S), k) * qk_scale
    grad_q += torch.einsum("bij,bjd->bid", W_factor, k) * qk_scale
    grad_q = grad_q - omega * DM + tau * r
    grad_q += (W * relmm_DM).sum(dim=-1, keepdim=True) * r
    grad_q -= (W * relmm_R).sum(dim=-1, keepdim=True) * U

    grad_k = torch.einsum("bij,bid->bjd", (G * S), q) * qk_scale
    grad_k += torch.einsum("bij,bid->bjd", W_factor, q) * qk_scale
    grad_k -= torch.einsum("bij,bid->bjd", (G * W / d), r)
    grad_k += torch.einsum("bij,bid->bjd", W, DM)
    grad_k -= torch.einsum("bij,bid->bjd", W * relmm_DM, r)
    grad_k += torch.einsum("bij,bid->bjd", W * relmm_R, U)

    grad_v = torch.einsum("bij,bid->bjd", S, grad_o)

    return grad_q, grad_k, grad_v
