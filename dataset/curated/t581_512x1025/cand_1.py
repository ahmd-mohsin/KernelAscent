import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 581
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    D,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 math, matching PyTorch's fp32 accumulation for bf16) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask,
                other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)          # masked lanes: exp(-inf)=0
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 (softmax output dtype), then re-read as fp32 for RMS
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- RMS norm in fp32 ----
    msq = tl.sum(tl.where(mask, p * p, 0.0), axis=0) / D
    r = tl.math.rsqrt(msq + 1e-6)
    y = (p * r).to(tl.bfloat16).to(tl.float32)

    # ---- scale by weight (bf16 mul uses fp32 opmath in PyTorch) ----
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
        _softmax_rms_kernel[(rows,)](
            x2, self.rms1_w, y,
            d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
