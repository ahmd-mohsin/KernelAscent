import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 764
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, G3, B3, G4, B4,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # scale (compute fp32, round to bf16)
    x = x * 1.3875
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu (erf-based, fp32, round to bf16)
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 1 (fp32 stats, output rounded to bf16)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 2
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g4 + b4

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y * 1.3875
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, y, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
