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
    X, G, B, Y,
    N, stride,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, biased variance, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # emulate fp16 round-trip between ops (matches reference dtype flow)
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # GELU (exact, erf-based)
    out = p * 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y + row * stride + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_ln_softmax_gelu[(Mrows,)](
            h, self.ln2_g, self.ln2_b, out,
            N, h.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
