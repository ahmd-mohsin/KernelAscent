import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 934
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_relu_ln_softmax_gelu(
    X, OUT, G, B,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # ReLU (output would be fp16, exact in fp16 anyway)
    x = tl.maximum(x, 0.0).to(tl.float16)

    # LayerNorm in fp32 (matches PyTorch opmath for half)
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    y = y.to(tl.float16)  # layer_norm output cast to half

    # Softmax in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)  # softmax output cast to half

    # GELU (exact, erf) in fp32
    z = sm.to(tl.float32)
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    out = out.to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_softmax_gelu[(Mrows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
