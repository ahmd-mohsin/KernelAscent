import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 919
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _bias_bias_scale_softmax(
    X, B1, B2, Y,
    N, stride_x, stride_y,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)

    # replicate fp16 rounding of each elementwise op (opmath fp32, round to fp16)
    t = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) * scale).to(tl.float16)

    tf = tl.where(mask, t.to(tl.float32), float('-inf'))
    m = tl.max(tf, 0)
    e = tl.exp(tf - m)
    s = tl.sum(e, 0)
    y = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _bias_bias_scale_softmax[(Mrows,)](
            h, self.b1, self.b2, out,
            N, h.stride(0), out.stride(0),
            1.4531,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
