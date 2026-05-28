import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def get_causal_mask(
    row_offset,
    col_offset,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    row_indices = row_offset + tl.arange(0, ROW_TILE_SIZE)
    col_indices = col_offset + tl.arange(0, COL_TILE_SIZE)
    mask = row_indices[:, None] >= col_indices[None, :]
    return mask


@triton.jit
def update_active_mask(norm0, norm, cg_atol, cg_rtol):
    tol_ref = tl.maximum(cg_atol, norm0)
    converged = norm <= (tol_ref * cg_rtol)
    invalid = libdevice.isnan(norm) | libdevice.isinf(norm)  # type: ignore
    finished = converged | invalid
    return ~finished


@triton.jit
def _cg_kernel(
    Qr,
    Xr,
    Yr,
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
    HEAD_DIM: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    Rr = Yr
    Pr = Yr
    norm0 = tl.sum(Rr * Rr, axis=1, keep_dims=True)
    active_mask = tl.full((ROW_TILE_SIZE, 1), 1, dtype=tl.int1)
    active_mask = active_mask & (norm0 != 0)
    ridge_lambda = tl.reshape(lr, (ROW_TILE_SIZE, 1)) * omegar
    NUM_COL_BLOCKS = tl.cdiv(
        tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE),
        COL_TILE_SIZE,
    )

    for _ in range(cg_max_iters):
        SigmaPr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

        for col_block_id in range(NUM_COL_BLOCKS):
            col_offset = col_block_id * COL_TILE_SIZE

            Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
            causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)
            PrKT = tl.dot(Pr.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32)
            qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale
            qk = tl.where(causal_mask, qk, -float('inf'))
            W = tl.math.exp2(qk - mr)
            WPrKT = (W * PrKT).to(tl.bfloat16)
            SigmaPr = tl.dot(WPrKT, Kc, out_dtype=tl.float32, acc=SigmaPr)

            k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))

        k_block_ptr = tl.advance(k_block_ptr, (-NUM_COL_BLOCKS * COL_TILE_SIZE, 0))

        Qr_fp32 = Qr.to(tl.float32)
        PQ = tl.sum(Pr * Qr_fp32, axis=1, keep_dims=True)
        PM = tl.sum(Pr * Mr, axis=1, keep_dims=True)
        SigmaPr = SigmaPr - PQ * Mr - PM * Qr_fp32 + omegar * PQ * Qr_fp32 + Pr * ridge_lambda
        norm = tl.sum(Rr * Rr, axis=1, keep_dims=True)
        denorm = tl.sum(Pr * SigmaPr, axis=1, keep_dims=True)
        alpha = tl.where(active_mask, norm / (denorm + 1e-10), 0.0)
        Xr = Xr + alpha * Pr
        Rr = Rr - alpha * SigmaPr
        norm_new = tl.sum(Rr * Rr, axis=1, keep_dims=True)
        active_mask = active_mask & update_active_mask(norm0, norm_new, cg_atol, cg_rtol)
        beta = tl.where(active_mask, norm_new / norm, 0.0)
        Pr = Rr + beta * Pr

    return Xr


@triton.jit
def _jacobi_preconditioned_cg_kernel(
    Qr,
    Xr,
    Yr,
    Mr,
    mr,
    lr,
    omegar,
    Sigma_diag,
    k_block_ptr,
    qk_scale,
    cg_atol,
    cg_rtol,
    cg_max_iters,
    row_offset,
    N_KEYVALS,
    HEAD_DIM: tl.constexpr,
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
):
    ridge_lambda = tl.reshape(lr, (ROW_TILE_SIZE, 1)) * omegar
    Sigma_diag = Sigma_diag + ridge_lambda
    Rr = Yr
    Zr = Rr / Sigma_diag
    Pr = Zr
    norm0 = tl.sum(Rr * Zr, axis=1, keep_dims=True)
    active_mask = tl.full((ROW_TILE_SIZE, 1), 1, dtype=tl.int1)
    active_mask = active_mask & (norm0 != 0)
    NUM_COL_BLOCKS = tl.cdiv(
        tl.minimum(N_KEYVALS, row_offset + ROW_TILE_SIZE),
        COL_TILE_SIZE,
    )

    for _ in range(cg_max_iters):
        SigmaPr = tl.zeros((ROW_TILE_SIZE, HEAD_DIM), dtype=tl.float32)

        for col_block_id in range(NUM_COL_BLOCKS):
            col_offset = col_block_id * COL_TILE_SIZE

            Kc = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option='zero')
            causal_mask = get_causal_mask(row_offset, col_offset, ROW_TILE_SIZE, COL_TILE_SIZE)
            PrKT = tl.dot(Pr.to(tl.bfloat16), tl.trans(Kc), out_dtype=tl.float32)
            qk = tl.dot(Qr, tl.trans(Kc), out_dtype=tl.float32) * qk_scale
            qk = tl.where(causal_mask, qk, -float('inf'))
            W = tl.math.exp2(qk - mr)
            WPrKT = (W * PrKT).to(tl.bfloat16)
            SigmaPr = tl.dot(WPrKT, Kc, out_dtype=tl.float32, acc=SigmaPr)

            k_block_ptr = tl.advance(k_block_ptr, (COL_TILE_SIZE, 0))

        k_block_ptr = tl.advance(k_block_ptr, (-NUM_COL_BLOCKS * COL_TILE_SIZE, 0))

        Qr_fp32 = Qr.to(tl.float32)
        PQ = tl.sum(Pr * Qr_fp32, axis=1, keep_dims=True)
        PM = tl.sum(Pr * Mr, axis=1, keep_dims=True)
        SigmaPr = SigmaPr - PQ * Mr - PM * Qr_fp32 + omegar * PQ * Qr_fp32 + Pr * ridge_lambda
        norm = tl.sum(Rr * Zr, axis=1, keep_dims=True)
        denorm = tl.sum(Pr * SigmaPr, axis=1, keep_dims=True)
        alpha = tl.where(active_mask, norm / (denorm + 1e-10), 0.0)
        Xr = Xr + alpha * Pr
        Rr = Rr - alpha * SigmaPr
        Zr = Rr / Sigma_diag
        norm_new = tl.sum(Rr * Zr, axis=1, keep_dims=True)
        active_mask = active_mask & update_active_mask(norm0, norm_new, cg_atol, cg_rtol)
        beta = tl.where(active_mask, norm_new / norm, 0.0)
        Pr = Zr + beta * Pr

    return Xr
