import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 540
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, B1, B2, OUT,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32), then round to bf16 like reference
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # bf16 elementwise ops: compute in fp32, round to bf16 each step (matches PyTorch opmath)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32, round to bf16 (matches F.gelu on bf16)
    yf = y.to(tl.float32)
    g = (yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch bf16 softmax which upcasts)
    gf = tl.where(mask, g.to(tl.float32), float('-inf'))
    mx = tl.max(gf, axis=0)
    e = tl.exp(gf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.dim() == 2
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.rms0_w, self.b1, self.b2, out,
            n, x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
