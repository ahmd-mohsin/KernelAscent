import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 740
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, W_rms, G, B, Out,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 math, then cast to bf16 before weight multiply)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W_rms + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y_bf = xn * w  # bf16 multiply, matches reference dtype semantics
    y = y_bf.to(tl.float32)

    # Softmax in fp32, then round to bf16 (matches torch.softmax on bf16)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p_bf = (e / s).to(tl.bfloat16)
    p = p_bf.to(tl.float32)

    # LayerNorm in fp32 (matches F.layer_norm on bf16 input)
    mean = tl.sum(p, axis=0) / N
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (p - mean) * rstd * g + b

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_ln_kernel[(Mrows,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
