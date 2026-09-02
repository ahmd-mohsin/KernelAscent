import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 859
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_rms_softmax_rms_relu(
    X, W0, W3, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 (compute in fp32, round to fp16, multiply weight) ----
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (x * inv).to(tl.float16).to(tl.float32) * w0
    h = h.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, fp16 output) ----
    h = tl.where(mask, h, float('-inf'))
    mval = tl.max(h, axis=0)
    e = tl.exp(h - mval)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # ---- Scale ----
    p = (p * 1.2053).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms2 = tl.sum(p * p, axis=0) / D
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    o = (p * inv2).to(tl.float16).to(tl.float32) * w3
    o = o.to(tl.float16)

    # ---- ReLU ----
    o = tl.maximum(o, tl.zeros_like(o))

    tl.store(Y + row * stride_y + offs, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_rms_softmax_rms_relu[(m,)](
            x2, self.rms0_w, self.rms3_w, y,
            d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
