import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 891
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _softmax_ln_kernel(
    X, Y, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # cast to fp16 (softmax output dtype), then layernorm in fp32
    s16 = s.to(tl.float16)
    sf = s16.to(tl.float32)

    mean = tl.sum(sf, axis=0) / N
    d = tl.where(mask, sf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (sf - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_ln_kernel[(rows,)](
            h, y, self.ln2_g, self.ln2_b,
            h.stride(0), y.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
