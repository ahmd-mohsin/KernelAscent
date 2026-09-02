import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 266
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_epilogue_ln(
    X, B1, B2, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # match PyTorch: half adds computed at fp32 opmath, rounded to fp16 each step
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    # relu
    x = tl.maximum(x, tl.zeros_like(x))
    # scalar mul at fp32 opmath, rounded to fp16
    x = (x.to(tl.float32) * SCALE).to(tl.float16)

    # layer norm in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x @ self.W0
            x = x + self.b1
            x = x + self.b2
            x = torch.relu(x)
            x = x * 1.0466
            return F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)

        h = x @ self.W0  # cuBLAS fp16 tensor-core matmul
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_ln[(m,)](
            h, self.b1, self.b2, self.ln5_g, self.ln5_b, out,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, SCALE=1.0466,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
