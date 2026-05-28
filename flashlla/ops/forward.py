from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from flashlla.ops.cg import (
    _cg_kernel,
    _jacobi_preconditioned_cg_kernel,
    get_causal_mask,
)


_CONFIGS = [
    triton.Config({"ROW_TILE_SIZE": 64,  "COL_TILE_SIZE": 64},  num_warps=8, num_stages=3),
    triton.Config({"ROW_TILE_SIZE": 64,  "COL_TILE_SIZE": 128}, num_warps=8, num_stages=4),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 64},  num_warps=8, num_stages=4),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 128}, num_warps=8, num_stages=4),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 256}, num_warps=8, num_stages=4),
    triton.Config({"ROW_TILE_SIZE": 256, "COL_TILE_SIZE": 128}, num_warps=8, num_stages=4),
    triton.Config({"ROW_TILE_SIZE": 256, "COL_TILE_SIZE": 256}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_CONFIGS, key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM"])
@triton.jit
def _lla_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    r_ptr,
    d_ptr,
    m_ptr,
    l_ptr,
    w_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_rb, stride_rq, stride_rd,
    stride_db, stride_dq,
    stride_mb, stride_mq,
    stride_lb, stride_lq,
    stride_wb, stride_wq,
    qk_scale,
    delta_eps,
    cg_atol,
    cg_rtol,
    cg_max_iters,
    BATCH_SIZE,
    N_QUERIES,
    N_KEYVALS,
    USE_PRECONDITIONER: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid_row = tl.program_id(0)
    row_offset = pid_row * ROW_TILE_SIZE
    # Skip column tiles entirely in the future: their qk is -inf after the
    # causal mask, contributing zero to every accumulator.
    NUM_COL_BLOCKS = tl.cdiv(
        tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE),
        COL_TILE_SIZE,
    )

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_qq, stride_qd),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + pid_batch * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + pid_batch * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + pid_batch * stride_ob,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_oq, stride_od),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_rq, stride_rd),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + pid_batch * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq, 1),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_mb,
        shape=(N_QUERIES, 1),
        strides=(stride_mq, 1),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + pid_batch * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr + pid_batch * stride_wb,
        shape=(N_QUERIES, 1),
        strides=(stride_wq, 1),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )

    Qr = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    lr = tl.load(l_block_ptr, boundary_check=(0, 1), padding_option='zero')
    Qr_fp32 = Qr.to(tl.float32)
    mr = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32) - float('inf')
    omegar = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    Mr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    Rr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    Or = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    qk_scale = qk_scale * 1.44269504  # 1 / log(2), folded in so we can use exp2

    if USE_PRECONDITIONER:
        Dr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

    for col_block_id in range(NUM_COL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)
        qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale
        qk = tl.where(causal_mask, qk, -float('inf'))
        m = tl.max(qk, axis=1, keep_dims=True)
        m = tl.maximum(mr, m)
        W = tl.math.exp2(qk - m)

        alpha = tl.math.exp2(mr - m)
        omegar = alpha * omegar + tl.sum(W, axis=1, keep_dims=True)
        Mr = alpha * Mr
        Mr = tl.dot(W.to(tl.bfloat16), Kc, out_dtype=tl.float32, acc=Mr)
        if USE_PRECONDITIONER:
            Dr = alpha * Dr
            Dr = tl.dot(W.to(tl.bfloat16), (Kc * Kc), out_dtype=tl.float32, acc=Dr)
        mr = m

        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))

    k_block_ptr = tl.advance(k_block_ptr, (-NUM_COL_BLOCKS * COL_TILE_SIZE, 0))

    if USE_PRECONDITIONER:
        Dr = Dr - 2.0 * (Mr * Qr_fp32) + omegar * (Qr_fp32 * Qr_fp32)
        Rr = _jacobi_preconditioned_cg_kernel(
            Qr,
            Rr,
            Mr - omegar * Qr_fp32,
            Mr,
            mr,
            lr,
            omegar,
            Dr,
            k_block_ptr,
            qk_scale,
            cg_atol,
            cg_rtol,
            cg_max_iters,
            row_offset,
            N_KEYVALS,
            HEAD_DIM,
            ROW_TILE_SIZE,
            COL_TILE_SIZE,
        )
    else:
        Rr = _cg_kernel(
            Qr,
            Rr,
            Mr - omegar * Qr_fp32,
            Mr,
            mr,
            lr,
            omegar,
            k_block_ptr,
            qk_scale,
            cg_atol,
            cg_rtol,
            cg_max_iters,
            row_offset,
            N_KEYVALS,
            HEAD_DIM,
            ROW_TILE_SIZE,
            COL_TILE_SIZE,
        )

    Mr = Mr - omegar * Qr_fp32
    dr = omegar - tl.sum(Mr * Rr, axis=1, keep_dims=True)
    dr = dr + delta_eps * tl.where(dr >= 0, 1.0, -1.0)
    rq = (Rr * Qr_fp32).sum(axis=1, keep_dims=True)

    for col_block_id in range(NUM_COL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE

        Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Vc = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')
        causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)

        qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale
        qk = tl.where(causal_mask, qk, -float('inf'))
        W = tl.math.exp2(qk - mr)

        S_term = 1.0 - tl.dot(Rr.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32) + rq
        Or = tl.dot((W * S_term / dr).to(tl.bfloat16), Vc, acc=Or, out_dtype=tl.float32)

        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    tl.store(o_block_ptr, Or.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(r_block_ptr, Rr.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(d_block_ptr, dr, boundary_check=(0, 1))
    tl.store(m_block_ptr, mr, boundary_check=(0, 1))
    tl.store(w_block_ptr, omegar, boundary_check=(0, 1))


def lla_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ridge_lambda: torch.Tensor,
    qk_scale: float | torch.Tensor,
    delta_eps: float = 1e-12,
    cg_atol: float = 1e-12,
    cg_rtol: float = 1e-12,
    cg_max_iters: int = 32,
    use_preconditioner: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Triton LLA forward kernel.

    Args:
        q, k, v: ``(batch, seqlen, head_dim)`` tensors. The kernel expects
            bfloat16 inputs; callers are responsible for casting.
        ridge_lambda: per-row regularization. Broadcast to
            ``(batch, seqlen, 1)`` before launch.
        qk_scale: attention scale, typically ``1 / sqrt(head_dim)``.
        delta_eps: denominator floor for the LL weights.
        cg_atol, cg_rtol, cg_max_iters: CG solver tolerances.
        use_preconditioner: if True, use the Jacobi-preconditioned CG.

    Returns:
        Tuple ``(o, r, d, m, w)`` where ``o`` is the attention output,
        ``w`` is the per-query softmax normalizer (``omega``), and
        ``(r, d, m)`` are the intermediates needed by the backward pass.
    """

    BATCH_SIZE, N_QUERIES, HEAD_DIM = q.shape
    N_KEYVALS = k.shape[1]

    o = torch.empty((BATCH_SIZE, N_QUERIES, HEAD_DIM), device=q.device, dtype=q.dtype)
    r = torch.empty((BATCH_SIZE, N_QUERIES, HEAD_DIM), device=q.device, dtype=q.dtype)
    m = torch.empty((BATCH_SIZE, N_QUERIES, 1), device=q.device, dtype=torch.float32)
    d = torch.empty((BATCH_SIZE, N_QUERIES, 1), device=q.device, dtype=torch.float32)
    l = ridge_lambda.expand(BATCH_SIZE, N_QUERIES).unsqueeze(-1)
    w = torch.empty((BATCH_SIZE, N_QUERIES, 1), device=q.device, dtype=torch.float32)

    grid = lambda META: (math.ceil(N_QUERIES / META['ROW_TILE_SIZE']), BATCH_SIZE)

    _lla_fwd_kernel[grid](
        q, k, v,
        o, r, d, m, l, w,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        d.stride(0), d.stride(1),
        m.stride(0), m.stride(1),
        l.stride(0), l.stride(1),
        w.stride(0), w.stride(1),
        qk_scale, delta_eps, cg_atol, cg_rtol, cg_max_iters,
        BATCH_SIZE, N_QUERIES, N_KEYVALS,
        use_preconditioner,
        HEAD_DIM,
    )

    return o, r, d, m, w
