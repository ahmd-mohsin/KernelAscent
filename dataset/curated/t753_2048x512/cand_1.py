import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 753
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_gelu_ln_bias(
    X, G, B, B3, Y,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 to match reference intermediate precision
    g = g.to(tl.bfloat16).to(tl.float32)

    # layer norm (stats in fp32, matching PyTorch)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (g - mean) * rstd * w + b
    # layer_norm output is bf16 in reference
    y = y.to(tl.bfloat16).to(tl.float32)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y + b3).to(tl.bfloat16)

    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores)
        h = torch.matmul(x, self.W0)

        m, n = h.shape
        h = h.contiguous()
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_ln_bias[(m,)](
            h, self.ln2_g, self.ln2_b, self.b3, y,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
