import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 131
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _bias_softmax_ln_kernel(
    X, B, G, Bt, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    # bias add in fp16 (matches reference: x + b1 in fp16)
    t = (x + b).to(tl.float32)

    # softmax in fp32
    t_masked = tl.where(mask, t, float('-inf'))
    m = tl.max(t_masked, axis=0)
    e = tl.exp(t_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)  # reference softmax outputs fp16

    # layernorm in fp32 on fp16 softmax output
    v = sm.to(tl.float32)
    mean = tl.sum(v, axis=0) / N
    d = tl.where(mask, v - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    out = (v - mean) * rstd * g + bt
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


@triton.jit
def _ln_kernel(
    X, G, Bt, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x - mean) * rstd * g + bt
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 4096) fp16

        Mrows, N = h.shape
        BLOCK = triton.next_power_of_2(N)

        t = torch.empty_like(h)
        _bias_softmax_ln_kernel[(Mrows,)](
            h, self.b1, self.ln3_g, self.ln3_b, t,
            N, h.stride(0), t.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2
        h2 = t @ self.W4  # (M, 4096) fp16

        out = torch.empty_like(h2)
        _ln_kernel[(Mrows,)](
            h2, self.ln5_g, self.ln5_b, out,
            N, h2.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
