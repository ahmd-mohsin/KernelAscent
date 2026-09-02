import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 784
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X, B2, G3, Bt3, G4, Bt4, Out,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # relu + bias, round to bf16 to match reference elementwise semantics
    x = tl.maximum(x, 0.0)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    bt3 = tl.load(Bt3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g3 + bt3
    y = y.to(tl.bfloat16).to(tl.float32)  # intermediate rounded to bf16

    # LayerNorm 2
    mean2 = tl.sum(y, axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    bt4 = tl.load(Bt4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g4 + bt4

    tl.store(Out + row * stride_o + cols, z.to(tl.bfloat16), mask=mask)


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
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 16 if BLOCK >= 4096 else 8
        _fused_epilogue_kernel[(m,)](
            h, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            n, h.stride(0), out.stride(0),
            EPS=1e-5, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
