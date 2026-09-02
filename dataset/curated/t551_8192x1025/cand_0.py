import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 551
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_rms_bias_dsoftmax(
    X, W, B, OUT,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (compute in fp32, like reference) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn16 = (x * inv).to(tl.float16)  # cast to fp16, like .to(x.dtype)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # fp16 multiply then fp16 add, each rounded separately (matches PyTorch).
    # Products/sums of fp16 values are exact in fp32, so round-tripping via
    # fp32 with explicit fp16 rounding reproduces fp16 arithmetic exactly
    # and prevents unwanted FMA contraction.
    t = (xn16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    h = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # ---- softmax #1 (fp32 accumulation, fp16 output, like PyTorch) ----
    hf = tl.where(mask, h.to(tl.float32), float('-inf'))
    m1 = tl.max(hf, axis=0)
    e1 = tl.exp(hf - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.float16)

    # ---- softmax #2 ----
    pf = tl.where(mask, p1.to(tl.float32), float('-inf'))
    m2 = tl.max(pf, axis=0)
    e2 = tl.exp(pf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.reshape(-1, d).contiguous()
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_rms_bias_dsoftmax[(m,)](
            x2, self.rms0_w, self.b1, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
