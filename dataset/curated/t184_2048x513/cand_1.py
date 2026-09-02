import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 184
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_softmax_scale_relu_ln(
    X_ptr, G_ptr, B_ptr, Y_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # scale (opmath fp32, result bf16) then relu
    v = (p.to(tl.float32) * SCALE).to(tl.bfloat16)
    v = tl.maximum(v, 0.0).to(tl.bfloat16)

    # layernorm in fp32
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, 0.0)
    mean = tl.sum(vf, axis=0) / N
    d = tl.where(mask, vf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    tl.store(Y_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_softmax_scale_relu_ln[(Mrows,)](
            h, self.ln4_g, self.ln4_b, y,
            N=N, BLOCK=triton.next_power_of_2(N), EPS=1e-5, SCALE=1.2551,
            num_warps=8,
        )
        return y
