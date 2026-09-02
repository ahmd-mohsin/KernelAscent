import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 104
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _bias_ln_bias_kernel(
    Y, B1, B2, B3, G, B, B5, OUT,
    N, stride_row, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride_row + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # replicate sequential fp16 additions
    t = y + b1
    t = t + b2
    t = t + b3

    x = t.to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    out = (x - mean) * rstd * g + bb
    out16 = out.to(tl.float16)

    b5 = tl.load(B5 + cols, mask=mask, other=0.0)
    out16 = out16 + b5

    tl.store(OUT + row * stride_row + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x + self.b1
            x = x + self.b2
            x = x + self.b3
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            x = x + self.b5
            return x

        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        out = torch.empty_like(y)
        n_rows = y.shape[0]
        N = y.shape[-1]
        BLOCK = triton.next_power_of_2(N)
        _bias_ln_bias_kernel[(n_rows,)](
            y, self.b1, self.b2, self.b3,
            self.ln4_g, self.ln4_b, self.b5, out,
            N, y.stride(0), 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
