# FlashLLA

Efficient Triton kernels for **Local Linear Attention (LLA)** — the
attention mechanism introduced in:

> **Local Linear Attention: An Optimal Interpolation of Linear and Softmax
> Attention For Test-Time Regression.**
> Yifei Zuo, Yutong Yin, Zhichen Zeng, Ang Li, Banghua Zhu, Zhaoran Wang.
> *ICLR 2026.*
> [[arXiv]](https://arxiv.org/abs/2510.01450)
> [[OpenReview]](https://openreview.net/forum?id=WGpzi489XY)

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

`flashlla` exposes two entry points; both are causal.

### `lla_attention(q, k, v, *, ridge_lambda=0.01, qk_scale=None, ...) -> Tensor`

The recommended functional entry point. Returns the attention output `o`
with shape `(batch, seqlen, head_dim)`.

- `q, k, v` — CUDA tensors of shape `(B, S, D)`. Inputs in fp16 / bf16 /
  fp32 are accepted; the kernel runs in bf16 internally.
- `ridge_lambda` — scalar `float` or a broadcastable tensor controlling
  the ridge-regression regularizer. This is the linear↔softmax
  interpolation knob.
- `qk_scale` — defaults to `1 / sqrt(head_dim)`.
- CG solver controls (`cg_atol`, `cg_rtol`, `cg_max_iters`,
  `cg_use_preconditioner`) tune the inner ridge solve. Defaults are
  usually fine; flip `cg_use_preconditioner=True` for ill-conditioned
  contexts.

### `LLAFunction.apply(q, k, v, ridge_lambda, qk_scale, delta_eps, cg_atol, cg_rtol, cg_max_iters, cg_use_preconditioner) -> (o, omega)`

The raw `torch.autograd.Function`, returning both the output and the
normalizer `omega` (useful for analysis and downstream losses that need
the per-position regression weight).

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
