import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 590
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, B3, Y, N, eps,
                  BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    # x * 1.254 rounded to bf16 (match original bf16 op)
    x = (x.to(tl.float32) * 1.254).to(tl.bfloat16)
    # relu
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)
    # layernorm with fp32 accumulation (as PyTorch does for bf16)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd * g + b).to(tl.bfloat16)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    y = (y.to(tl.float32) * 1.1001).to(tl.bfloat16)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.ln2_g, self.ln2_b, self.b3, y,
            N, 1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y.view(orig_shape)
