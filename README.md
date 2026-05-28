# FlashLLA

[![arXiv](https://img.shields.io/badge/arXiv-2510.01450-b31b1b.svg)](https://arxiv.org/abs/2510.01450)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

Efficient Triton kernels for **Local Linear Attention (LLA)** — the
attention mechanism introduced in:

> **Local Linear Attention: An Optimal Interpolation of Linear and Softmax
> Attention For Test-Time Regression.**
> Yifei Zuo, Yutong Yin, Zhichen Zeng, Ang Li, Banghua Zhu, Zhaoran Wang.
> *ICLR 2026.*

LLA performs a local linear estimate over the running KV context at every query position, recovering both the kernel regression view of Softmax Attention and the OLS solve in Linear Attention (MesaNet) as limiting cases. FlashLLA provides a fused, causal forward/backward implementation of the operator with a built-in conjugate gradient solver for the inner ridge regression.

## Install

```bash
git clone https://github.com/Yifei-Zuo/FlashLLA.git
cd FlashLLA
pip install -e .
```

## Quickstart

```python
import torch
from flashlla import lla_attention

B, S, D = 2, 1024, 64
q = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
o = lla_attention(q, k, v, ridge_lambda=0.1)   # (B, S, D)

o.float().pow(2).mean().backward()
```

A self-contained runnable demo with both the functional API and the
explicit `LLAFunction` (which exposes the inner normalizer `omega`) is in
[`examples/minimal_usage.py`](examples/minimal_usage.py).

## API

`flashlla` exposes two causal entry points.

### `lla_attention` — functional API

```python
lla_attention(
    q, k, v,
    *,
    ridge_lambda=10.0,
    qk_scale=None,
    delta_eps=1e-12,
    cg_atol=1e-12,
    cg_rtol=1e-12,
    cg_max_iters=32,
    cg_use_preconditioner=False,
) -> Tensor  # (B, S, D)
```

| Argument | Description |
|---|---|
| `q, k, v` | CUDA tensors of shape `(B, S, D)`. The kernel runs in bf16 internally. |
| `ridge_lambda` | Scalar `float` or broadcastable tensor. |
| `qk_scale` | Defaults to `1 / sqrt(head_dim)`. |
| `cg_*` controls | Inner conjugate-gradient solver tolerances and iteration cap. Use `cg_use_preconditioner=True` for a lower maximal iteration count. |

### `LLAFunction` — raw `autograd.Function`

```python
o, omega = LLAFunction.apply(
    q, k, v,
    ridge_lambda,           # tensor matching q's device/dtype
    qk_scale,               # float
    delta_eps,              # float
    cg_atol, cg_rtol,       # float
    cg_max_iters,           # int
    cg_use_preconditioner,  # bool
)
```

## Numerical notes

- **Internal precision**: matmuls run in bf16 with fp32 accumulation;
  inputs are cast as needed.
- **Inner solve**: each query position runs a small ridge regression via
  a fused conjugate-gradient solver (with an optional Jacobi preconditioner).
- **Reference impl**: `flashlla.ops.naive.lla_forward_naive` (and its
  backward counterpart) implement the same math in pure fp32 PyTorch
  using `torch.linalg.solve`.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

Tests require a CUDA GPU. They sweep small `(batch, seqlen, head_dim)`
configurations and compare against the fp32 naive reference.

## Citation

```bibtex
@inproceedings{zuo2025locallinear,
  title     = {Local Linear Attention: An Optimal Interpolation of Linear and Softmax Attention For Test-Time Regression},
  author    = {Zuo, Yifei and Yin, Yutong and Zeng, Zhichen and Li, Ang and Zhu, Banghua and Wang, Zhaoran},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=WGpzi489XY}
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
