import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 784
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_ln_ln(
    X, B2, G3, B3, G4, B4, Y,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # relu + bias (compute in fp32, round back to bf16 like PyTorch elementwise ops)
    x = tl.maximum(x, 0.0) + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (stats in fp32, matching PyTorch bf16 layer_norm)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * inv1 * g3 + b3

    # round to bf16 (output of first layer_norm), then LN2 in fp32
    y = y.to(tl.bfloat16).to(tl.float32)
    mean2 = tl.sum(y, axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * inv2 * g4 + b4

    tl.store(Y + row * stride + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape[0], h.shape[1]
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_bias_ln_ln[(rows,)](
            h, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            N, h.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
