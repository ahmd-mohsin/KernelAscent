import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 744
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_relu_scale_ln_ln(
    X, G3, B3, G4, B4, Out,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # relu + scale (fp16 opmath is fp32, result rounded to fp16)
    x = tl.maximum(x, 0.0) * 1.0687
    x = x.to(tl.float16).to(tl.float32)

    n = N.to(tl.float32)

    # LayerNorm 1 (stats in fp32, like PyTorch for fp16 input)
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / n
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n
    inv1 = 1.0 / tl.sqrt(var1 + EPS)

    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean1) * inv1 * g3 + b3
    # round to fp16 as the reference materializes an fp16 tensor between LNs
    y = y.to(tl.float16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n
    inv2 = 1.0 / tl.sqrt(var2 + EPS)

    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y - mean2) * inv2 * g4 + b4

    tl.store(Out + row * stride + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_scale_ln_ln[(rows,)](
            h, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, out,
            N, h.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
