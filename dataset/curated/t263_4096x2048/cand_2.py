import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 263
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_bias_softmax_ln_relu(
    X, B, G, Beta, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in bf16 (matches reference: bf16 + bf16 -> bf16)
    xb = (x + b).to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # softmax in fp32
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to bf16 (reference softmax outputs bf16 before layer_norm)
    p = p.to(tl.bfloat16).to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Beta + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + bt

    # cast to bf16, then relu (matches reference: relu applied to bf16)
    y = y.to(tl.bfloat16)
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    tl.store(Out + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_ln_relu[(Mrows,)](
            y, self.b1, self.ln3_g, self.ln3_b, out,
            y.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=16,
        )
        return out
