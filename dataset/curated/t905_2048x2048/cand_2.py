import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 905
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(X, W, Out, N, stride_x, stride_o,
                        BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 (matches reference: softmax output in bf16, then .float())
    p_bf = p.to(tl.bfloat16)
    pf = p_bf.to(tl.float32)

    # rmsnorm
    ms = tl.sum(pf * pf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (pf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = y * w
    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, out, N,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
