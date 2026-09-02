import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import triton.language.extra.libdevice as _libdevice
    _HAS_LIBDEVICE = True
except Exception:
    try:
        from triton.language.math import libdevice as _libdevice
        _HAS_LIBDEVICE = True
    except Exception:
        _libdevice = None
        _HAS_LIBDEVICE = False

SEED = 93
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_relu_softmax_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_row,
    HAS_LIBDEVICE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (fp16 -> fp32), relu
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # softmax in fp32 (matching PyTorch's fp32 accumulation for half inputs)
    row_max = tl.max(x, axis=0)
    shifted = x - row_max
    if HAS_LIBDEVICE:
        e = _libdevice.exp(shifted)
    else:
        e = tl.exp(shifted)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.float16)  # round to fp16 like reference softmax output

    # + bias (PyTorch opmath for half is fp32, result rounded to fp16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # RMSNorm in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    if HAS_LIBDEVICE:
        rinv = _libdevice.rsqrt(ms + 1e-6)
    else:
        rinv = 1.0 / tl.sqrt(ms + 1e-6)
    z = (yf * rinv).to(tl.float16)

    # * weight (fp32 opmath, rounded to fp16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = (z.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()
        rows = h2.shape[0]

        out = torch.empty_like(h2)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_softmax_bias_rms_kernel[(rows,)](
            h2, self.b3, self.rms4_w, out,
            N, h2.stride(0),
            HAS_LIBDEVICE=_HAS_LIBDEVICE,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.reshape(orig_shape)
