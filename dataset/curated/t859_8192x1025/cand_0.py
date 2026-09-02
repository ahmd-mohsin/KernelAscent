import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 859
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, w0_ptr, w3_ptr, out_ptr,
    n_cols,
    stride_row,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x_row = x_ptr + row * stride_row
    x = tl.load(x_row + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # ---- RMSNorm 1 (float compute, cast to fp16, fp16 mul with weight) ----
    ms = tl.sum(xf * xf, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + EPS)
    y16 = (xf * inv).to(tl.float16)
    w0 = tl.load(w0_ptr + cols, mask=mask, other=0.0)  # fp16
    y16 = y16 * w0  # fp16 arithmetic

    # ---- Softmax (float accumulation, like PyTorch half softmax) ----
    yf = y16.to(tl.float32)
    yf_m = tl.where(mask, yf, float("-inf"))
    mx = tl.max(yf_m, axis=0)
    e = tl.exp(yf_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)

    # ---- scale by 1.2053 (float opmath, cast back to fp16) ----
    z16 = (sm16.to(tl.float32) * SCALE).to(tl.float16)

    # ---- RMSNorm 2 ----
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    ms2 = tl.sum(zf * zf, axis=0) / n_cols
    inv2 = 1.0 / tl.sqrt(ms2 + EPS)
    o16 = (zf * inv2).to(tl.float16)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0)
    o16 = o16 * w3

    # ---- ReLU ----
    zero = tl.zeros(o16.shape, dtype=tl.float16)
    o16 = tl.maximum(o16, zero)

    tl.store(out_ptr + row * stride_row + cols, o16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, self.rms0_w, self.rms3_w, out,
            d, x.stride(0),
            EPS=1e-6, SCALE=1.2053,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
