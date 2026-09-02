import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 974
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, W_rms, G, B, Out,
    N, stride_x, stride_o,
    eps_rms, eps_ln, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to bf16, then bf16-style mul with weight)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = tl.math.rsqrt(ms + eps_rms)
    y = (xf * rrms).to(tl.bfloat16)

    w = tl.load(W_rms + cols, mask=mask, other=0.0)
    z = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32 accumulation (matches PyTorch acc_type behavior)
    zf = z.to(tl.float32)
    mean = tl.sum(tl.where(mask, zf, 0.0), axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((zf - mean) * rstd * g + b).to(tl.bfloat16)

    out = (ln.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norms_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            n, x.stride(0), out.stride(0),
            1e-6, 1e-5, 1.3244,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
