import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 287
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _ln_gelu_bias_kernel(
    X, G, B, B3, Y,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    # round to fp16 (layer_norm output dtype), then compute gelu in fp32 like PyTorch
    y = y.to(tl.float16).to(tl.float32)

    gel = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    gel = gel.to(tl.float16)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = gel + b3

    tl.store(Y + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _ln_gelu_bias_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, self.b3, out,
            n, h.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
