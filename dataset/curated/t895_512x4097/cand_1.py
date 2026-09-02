import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 895
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_ln_rms_gelu(X, G, B, W, Out, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, matching PyTorch)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # LN output rounded to fp16 (as in reference), then re-promoted for RMS
    y = (diff * rstd * g + b).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / D
    rrms = 1.0 / tl.sqrt(ms + 1e-6)

    w = tl.load(W + offs, mask=mask, other=0.0)
    # (yf * rrms).to(fp16) * w  (fp16 multiply == fp32 multiply + round)
    z = ((y * rrms).to(tl.float16).to(tl.float32) * w.to(tl.float32)).to(tl.float16).to(tl.float32)

    # Exact GELU (erf-based), fp32 opmath then cast to fp16
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Out + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_ln_rms_gelu[(m,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, out, d,
            BLOCK=BLOCK, num_warps=16,
        )
        return out.view(orig_shape)
