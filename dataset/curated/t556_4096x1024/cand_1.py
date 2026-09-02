import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 556
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_ln_kernel(
    X, OUT, G, B, B4,
    N, stride_xm, stride_om,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, as PyTorch does for bf16 inputs)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # round to bf16 (softmax output dtype in reference), relu is a no-op on >=0
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # layer norm with fp32 accumulation
    mean = tl.sum(sm, axis=0) / N
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (sm - mean) * rstd * g + b

    # cast to bf16, then add b4 in bf16 (matches reference bf16 add)
    y16 = y.to(tl.bfloat16)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    out = y16 + b4

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core matmul
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_softmax_ln_kernel[(m,)](
            h, out, self.ln3_g, self.ln3_b, self.b4,
            n, h.stride(0), out.stride(0),
            1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
