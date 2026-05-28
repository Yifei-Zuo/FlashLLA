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
    triton.Config({"ROW_TILE_SIZE": 64,  "COL_TILE_SIZE": 64},  num_warps=8, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 64,  "COL_TILE_SIZE": 128}, num_warps=8, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 64},  num_warps=8, num_stages=2),
    triton.Config({"ROW_TILE_SIZE": 128, "COL_TILE_SIZE": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM"])
@triton.jit
def _lla_bwd_q_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    r_ptr,
    d_ptr,
    m_ptr,
    l_ptr,
    grad_o_ptr,
    u_ptr,
    b_ptr,
    grad_q_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_rb, stride_rq, stride_rd,
    stride_db, stride_dq,
    stride_mb, stride_mq,
    stride_lb, stride_lq,
    stride_gob, stride_goq, stride_god,
    stride_gqb, stride_gqq, stride_gqd,
    stride_ub, stride_uq, stride_ud,
    stride_bb, stride_bq,
    qk_scale,
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
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_goq, stride_god),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    grad_q_block_ptr = tl.make_block_ptr(
        base=grad_q_ptr + pid_batch * stride_gqb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_gqq, stride_gqd),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    u_block_ptr = tl.make_block_ptr(
        base=u_ptr + pid_batch * stride_ub,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_uq, stride_ud),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1),
        strides=(stride_bq, 1),
        offsets=(row_offset, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )

    Qr = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    Rr = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option='zero')
    dr = tl.load(d_block_ptr, boundary_check=(0, 1), padding_option='zero')
    mr = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option='zero')
    lr = tl.load(l_block_ptr, boundary_check=(0, 1), padding_option='zero')
    GOr = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option='zero')

    Ur = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    GQr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

    betar = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    taur = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    omegar = tl.zeros((ROW_TILE_SIZE, 1), dtype=tl.float32)
    Tr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    Mr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

    rq = (Rr.to(tl.float32) * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)
    qk_scale_log2 = qk_scale * 1.44269504  # 1 / log(2), folded in so we can use exp2

    if USE_PRECONDITIONER:
        Dr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

    for col_block_id in range(NUM_COL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Vc = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')
        causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)

        qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(causal_mask, qk, -float('inf'))
        Gamma = tl.dot(GOr, tl.trans(Vc), out_dtype=tl.float32)

        W = tl.math.exp2(qk - mr)
        S_term = 1.0 - tl.dot(Rr, tl.trans(Kc), out_dtype=tl.float32) + rq
        S = S_term * W / dr
        C = Gamma * W

        betar += tl.sum(Gamma * S, axis=1, keep_dims=True)
        taur += tl.sum(C, axis=1, keep_dims=True)
        omegar += tl.sum(W, axis=1, keep_dims=True)

        Tr = tl.dot(C.to(tl.bfloat16), Kc, out_dtype=tl.float32, acc=Tr)
        Mr = tl.dot(W.to(tl.bfloat16), Kc, out_dtype=tl.float32, acc=Mr)
        if USE_PRECONDITIONER:
            Dr = tl.dot(W.to(tl.bfloat16), (Kc * Kc), out_dtype=tl.float32, acc=Dr)

        GQr = tl.dot(
            (Gamma * S).to(tl.bfloat16),
            (Kc * qk_scale).to(tl.bfloat16),
            out_dtype=tl.float32,
            acc=GQr,
        )

        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    k_block_ptr = tl.advance(k_block_ptr, (-COL_TILE_SIZE * NUM_COL_BLOCKS, 0))
    v_block_ptr = tl.advance(v_block_ptr, (-COL_TILE_SIZE * NUM_COL_BLOCKS, 0))

    betar = betar / dr
    Tr = Tr / dr
    taur = taur / dr
    Tr = Tr - taur * Qr.to(tl.float32)
    GQr += taur * Rr.to(tl.float32)

    if USE_PRECONDITIONER:
        Dr = Dr - 2.0 * (Mr * Qr.to(tl.float32)) + omegar * (Qr.to(tl.float32) * Qr.to(tl.float32))
        Ur = _jacobi_preconditioned_cg_kernel(
            Qr, Ur, Tr, Mr, mr, lr, omegar, Dr,
            k_block_ptr,
            qk_scale_log2,
            cg_atol, cg_rtol, cg_max_iters,
            row_offset, N_KEYVALS,
            HEAD_DIM, ROW_TILE_SIZE, COL_TILE_SIZE,
        )
    else:
        Ur = _cg_kernel(
            Qr, Ur, Tr, Mr, mr, lr, omegar,
            k_block_ptr,
            qk_scale_log2,
            cg_atol, cg_rtol, cg_max_iters,
            row_offset, N_KEYVALS,
            HEAD_DIM, ROW_TILE_SIZE, COL_TILE_SIZE,
        )

    DeltaMu = -Ur + 2 * betar * Rr.to(tl.float32)
    GQr -= omegar * DeltaMu
    uq = (Ur.to(tl.float32) * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)
    dmq = (DeltaMu.to(tl.float32) * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)

    for col_block_id in range(NUM_COL_BLOCKS):
        col_offset = col_block_id * COL_TILE_SIZE
        Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
        causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)

        qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale_log2
        qk = tl.where(causal_mask, qk, -float('inf'))
        W = tl.math.exp2(qk - mr)

        relmm_Rr = tl.dot(Rr, tl.trans(Kc), out_dtype=tl.float32) - rq
        relmm_Ur = tl.dot(Ur.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32) - uq
        relmm_DeltaMu = tl.dot(DeltaMu.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32) - dmq
        W_factor = W * (
            -betar + relmm_DeltaMu
            - 0.5 * relmm_DeltaMu * relmm_Rr
            + 0.5 * relmm_Ur * relmm_Rr
        )
        GQr = tl.dot(
            W_factor.to(tl.bfloat16),
            (Kc * qk_scale).to(tl.bfloat16),
            out_dtype=tl.float32,
            acc=GQr,
        )
        GQr += ((W * relmm_DeltaMu).sum(axis=1, keep_dims=True)) * Rr.to(tl.float32)
        GQr -= ((W * relmm_Rr).sum(axis=1, keep_dims=True)) * Ur.to(tl.float32)

        k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (COL_TILE_SIZE, 0))

    tl.store(grad_q_block_ptr, GQr.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(u_block_ptr, Ur.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(b_block_ptr, betar, boundary_check=(0, 1))


@triton.autotune(configs=_CONFIGS, key=["N_QUERIES", "N_KEYVALS", "HEAD_DIM"])
@triton.jit
def _lla_bwd_kv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    r_ptr,
    u_ptr,
    b_ptr,
    d_ptr,
    m_ptr,
    l_ptr,
    grad_o_ptr,
    grad_k_ptr,
    grad_v_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_rb, stride_rq, stride_rd,
    stride_ub, stride_uq, stride_ud,
    stride_bb, stride_bq,
    stride_db, stride_dq,
    stride_mb, stride_mq,
    stride_lb, stride_lq,
    stride_gob, stride_goq, stride_god,
    stride_gkb, stride_gkk, stride_gkd,
    stride_gvb, stride_gvk, stride_gvd,
    qk_scale,
    BATCH_SIZE,
    N_QUERIES,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid_col = tl.program_id(0)
    col_offset = pid_col * COL_TILE_SIZE
    NUM_ROW_BLOCKS = tl.cdiv(N_QUERIES, ROW_TILE_SIZE)
    # Skip row tiles entirely above this column tile (no causal overlap).
    FIRST_ROW_BLOCK = col_offset // ROW_TILE_SIZE

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + pid_batch * stride_qb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_qq, stride_qd),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + pid_batch * stride_kb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_kk, stride_kd),
        offsets=(col_offset, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + pid_batch * stride_vb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_vk, stride_vd),
        offsets=(col_offset, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    r_block_ptr = tl.make_block_ptr(
        base=r_ptr + pid_batch * stride_rb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_rq, stride_rd),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    u_block_ptr = tl.make_block_ptr(
        base=u_ptr + pid_batch * stride_ub,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_uq, stride_ud),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + pid_batch * stride_bb,
        shape=(N_QUERIES, 1),
        strides=(stride_bq, 1),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + pid_batch * stride_db,
        shape=(N_QUERIES, 1),
        strides=(stride_dq, 1),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    m_block_ptr = tl.make_block_ptr(
        base=m_ptr + pid_batch * stride_mb,
        shape=(N_QUERIES, 1),
        strides=(stride_mq, 1),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + pid_batch * stride_lb,
        shape=(N_QUERIES, 1),
        strides=(stride_lq, 1),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, 1),
        order=(1, 0),
    )
    grad_o_block_ptr = tl.make_block_ptr(
        base=grad_o_ptr + pid_batch * stride_gob,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_goq, stride_god),
        offsets=(FIRST_ROW_BLOCK * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    grad_k_block_ptr = tl.make_block_ptr(
        base=grad_k_ptr + pid_batch * stride_gkb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_gkk, stride_gkd),
        offsets=(col_offset, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    grad_v_block_ptr = tl.make_block_ptr(
        base=grad_v_ptr + pid_batch * stride_gvb,
        shape=(N_KEYVALS, HEAD_DIM),
        strides=(stride_gvk, stride_gvd),
        offsets=(col_offset, 0),
        block_shape=(COL_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
    Vc = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option='zero')

    GKc = tl.zeros((COL_TILE_SIZE, HEAD_DIM), dtype=tl.float32)
    GVc = tl.zeros((COL_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

    qk_scale_log2 = qk_scale * 1.44269504  # 1 / log(2), folded in so we can use exp2

    for row_block_id in range(FIRST_ROW_BLOCK, NUM_ROW_BLOCKS):
        row_offset = row_block_id * ROW_TILE_SIZE

        Qr = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Rr = tl.load(r_block_ptr, boundary_check=(0, 1), padding_option='zero')
        dr = tl.load(d_block_ptr, boundary_check=(0, 1), padding_option='zero')
        mr = tl.load(m_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Ur = tl.load(u_block_ptr, boundary_check=(0, 1), padding_option='zero')
        br = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
        GOr = tl.load(grad_o_block_ptr, boundary_check=(0, 1), padding_option='zero')

        qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale_log2
        causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)
        qk = tl.where(causal_mask, qk, -float('inf'))

        relmm_Rr = (
            tl.dot(Rr, tl.trans(Kc), out_dtype=tl.float32)
            - (Rr.to(tl.float32) * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)
        )
        W = tl.math.exp2(qk - mr)
        S_term = 1.0 - relmm_Rr
        S = (W * S_term / dr).to(tl.bfloat16)

        GVc = tl.dot(tl.trans(S), GOr, out_dtype=tl.float32, acc=GVc)
        Gamma = tl.dot(GOr, tl.trans(Vc), out_dtype=tl.float32)
        GKc = tl.dot(
            tl.trans(Gamma * S).to(tl.bfloat16),
            (Qr * qk_scale).to(tl.bfloat16),
            out_dtype=tl.float32,
            acc=GKc,
        )
        DeltaMu = -Ur.to(tl.float32) + 2 * br * Rr.to(tl.float32)
        relmm_Ur = (
            tl.dot(Ur, tl.trans(Kc), out_dtype=tl.float32)
            - (Ur.to(tl.float32) * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)
        )
        relmm_DeltaMu = (
            tl.dot(DeltaMu.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32)
            - (DeltaMu * Qr.to(tl.float32)).sum(axis=1, keep_dims=True)
        )

        W_factor = W * (
            -br + relmm_DeltaMu
            - 0.5 * relmm_DeltaMu * relmm_Rr
            + 0.5 * relmm_Ur * relmm_Rr
        )
        GKc = tl.dot(
            tl.trans(W_factor.to(tl.bfloat16)),
            (Qr * qk_scale).to(tl.bfloat16),
            out_dtype=tl.float32,
            acc=GKc,
        )
        GKc = tl.dot(
            tl.trans((Gamma * W) / dr).to(tl.bfloat16),
            -Rr,
            out_dtype=tl.float32,
            acc=GKc,
        )
        GKc = tl.dot(
            tl.trans(W.to(tl.bfloat16)),
            DeltaMu.to(tl.bfloat16),
            out_dtype=tl.float32,
            acc=GKc,
        )
        GKc = tl.dot(
            tl.trans(W * relmm_DeltaMu).to(tl.bfloat16),
            -Rr,
            out_dtype=tl.float32,
            acc=GKc,
        )
        GKc = tl.dot(
            tl.trans(W * relmm_Rr).to(tl.bfloat16),
            Ur,
            out_dtype=tl.float32,
            acc=GKc,
        )

        q_block_ptr = tl.advance(q_block_ptr, (ROW_TILE_SIZE, 0))
        r_block_ptr = tl.advance(r_block_ptr, (ROW_TILE_SIZE, 0))
        u_block_ptr = tl.advance(u_block_ptr, (ROW_TILE_SIZE, 0))
        b_block_ptr = tl.advance(b_block_ptr, (ROW_TILE_SIZE, 0))
        d_block_ptr = tl.advance(d_block_ptr, (ROW_TILE_SIZE, 0))
        m_block_ptr = tl.advance(m_block_ptr, (ROW_TILE_SIZE, 0))
        grad_o_block_ptr = tl.advance(grad_o_block_ptr, (ROW_TILE_SIZE, 0))

    tl.store(grad_k_block_ptr, GKc.to(tl.bfloat16), boundary_check=(0, 1))
    tl.store(grad_v_block_ptr, GVc.to(tl.bfloat16), boundary_check=(0, 1))


def lla_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    d: torch.Tensor,
    m: torch.Tensor,
    grad_o: torch.Tensor,
    ridge_lambda: torch.Tensor,
    qk_scale: float | torch.Tensor,
    cg_atol: float = 1e-8,
    cg_rtol: float = 1e-8,
    cg_max_iters: int = 40,
    use_preconditioner: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the Triton LLA backward kernels.

    The Q/K/V gradients are split across two kernels — ``_lla_bwd_q_kernel``
    parallelizes over query tiles and emits ``dQ`` plus the intermediates
    (``U``, ``beta``) that ``_lla_bwd_kv_kernel`` consumes to compute
    ``dK`` / ``dV``. The first query position has no causal context and
    therefore zero gradient; that position is explicitly zeroed on return
    to avoid spurious non-zero noise from the Triton accumulators.

    Returns:
        ``(grad_q, grad_k, grad_v)`` all in bfloat16.
    """

    BATCH_SIZE, N_QUERIES, HEAD_DIM = q.shape
    N_KEYVALS = k.shape[1]

    grad_q = torch.empty_like(q, dtype=torch.bfloat16)
    grad_k = torch.empty_like(k, dtype=torch.bfloat16)
    grad_v = torch.empty_like(v, dtype=torch.bfloat16)
    u = torch.empty_like(q, dtype=torch.bfloat16)
    b = torch.empty((BATCH_SIZE, N_QUERIES, 1), dtype=torch.float32, device=q.device)
    l = ridge_lambda.expand(BATCH_SIZE, N_QUERIES).unsqueeze(-1)

    q_grid = lambda META: (math.ceil(N_QUERIES / META['ROW_TILE_SIZE']), BATCH_SIZE)
    kv_grid = lambda META: (math.ceil(N_KEYVALS / META['COL_TILE_SIZE']), BATCH_SIZE)

    _lla_bwd_q_kernel[q_grid](
        q, k, v,
        r, d, m, l,
        grad_o,
        u, b,
        grad_q,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        d.stride(0), d.stride(1),
        m.stride(0), m.stride(1),
        l.stride(0), l.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        b.stride(0), b.stride(1),
        qk_scale,
        cg_atol, cg_rtol, cg_max_iters,
        BATCH_SIZE, N_QUERIES, N_KEYVALS,
        use_preconditioner,
        HEAD_DIM,
    )

    _lla_bwd_kv_kernel[kv_grid](
        q, k, v,
        r, u, b, d, m, l,
        grad_o,
        grad_k, grad_v,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        r.stride(0), r.stride(1), r.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        b.stride(0), b.stride(1),
        d.stride(0), d.stride(1),
        m.stride(0), m.stride(1),
        l.stride(0), l.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2),
        qk_scale,
        BATCH_SIZE, N_QUERIES, N_KEYVALS,
        HEAD_DIM,
    )

    grad_q[:, 0, :] = 0.0
    return grad_q, grad_k, grad_v
