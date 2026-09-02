import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 170
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_epilogue(
    X_ptr, Bias_ptr, G_ptr, Beta_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load matmul result row (fp16) and bias, add in fp32, round to fp16 (matches fp16 add)
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch on half), round result to fp16
    x_masked = tl.where(mask, x, float('-inf'))
    m = tl.max(x_masked, 0)
    e = tl.exp(x_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU (erf) computed in fp32 (opmath for half), rounded to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = p * 0.5 * (1.0 + tl.math.erf(p * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # layer norm in fp32 (matches PyTorch internal accumulation)
    g_m = tl.where(mask, g, 0.0)
    mean = tl.sum(g_m, 0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + beta

    tl.store(Out_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        y = torch.matmul(x, self.W0)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue[(m,)](
            y, self.b1, self.ln4_g, self.ln4_b, out,
            y.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
