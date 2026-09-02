import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 776
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_softmax_ln_rms_kernel(
    X, B1, LN_G, LN_B, RMS_W, OUT,
    D, stride_x, stride_o,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 compute, round to fp16 like PyTorch) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = (e / denom).to(tl.float16).to(tl.float32)

    # ---- add bias b1 (fp16 op, fp32 opmath) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = (sm + b1).to(tl.float16).to(tl.float32)

    # ---- layer norm (fp32 internal, eps=1e-5) ----
    x1m = tl.where(mask, x1, 0.0)
    mean = tl.sum(x1m, axis=0) / D
    diff = tl.where(mask, x1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- RMS norm (explicit fp32, eps=1e-6) ----
    ym = tl.where(mask, y, 0.0)
    ms = tl.sum(ym * ym, axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    z = (y * rrms).to(tl.float16).to(tl.float32)

    # ---- * rms3_w (fp16 op) then * 1.0801 (fp16 op) ----
    w = tl.load(RMS_W + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.float16).to(tl.float32)
    out = (z * scale).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.reshape(-1, d).contiguous()
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 8192 else 16
        _fused_softmax_ln_rms_kernel[(m,)](
            x2, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            d, x2.stride(0), out.stride(0),
            1.0801,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
