import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 808
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, out_ptr,
    n_cols,
    stride_x, stride_o,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # load input row (fp16), relu, upcast to fp32
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)  # cast to fp16 like reference

    # multiply by weight (opmath = fp32, round back to fp16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # relu (fp16)
    y = tl.maximum(y, y * 0)

    # scalar multiply (opmath fp32, round to fp16)
    y = (y.to(tl.float32) * scale).to(tl.float16)

    # softmax: upcast to fp32, compute, cast back to fp16
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.rms1_w, out,
            n_cols,
            x.stride(0), out.stride(0),
            1e-6,
            1.3505,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
