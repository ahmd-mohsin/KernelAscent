import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 549
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _ln_bias_gelu_kernel(
    X, G, B, B2, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # mean
    mean = tl.sum(x, axis=0) / N
    # variance (two-pass, biased)
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # round to bf16 (layer_norm output dtype), as reference does
    y_bf = y.to(tl.bfloat16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y_bf.to(tl.float32) + b2).to(tl.bfloat16)

    # exact GELU in fp32, then round to bf16
    zf = z.to(tl.float32)
    out = 0.5 * zf * (1.0 + tl.math.erf(zf * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _ln_bias_gelu_kernel[(m,)](
            h, self.ln1_g, self.ln1_b, self.b2, out,
            h.stride(0), out.stride(0),
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
