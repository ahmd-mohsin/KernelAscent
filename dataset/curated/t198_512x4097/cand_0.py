import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 198
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(X_ptr, W_ptr, Out_ptr, N, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # relu
    xf = tl.maximum(xf, 0.0)

    # rmsnorm
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    y_bf = (xf * rs).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    z_bf = y_bf * w  # bf16 multiply to match reference
    z = z_bf.to(tl.float32)

    # softmax 1 (float32 compute, bf16 round like torch)
    z = tl.where(mask, z, float('-inf'))
    m1 = tl.max(z, axis=0)
    e1 = tl.exp(z - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.bfloat16)

    # softmax 2
    p1f = tl.where(mask, p1.to(tl.float32), float('-inf'))
    m2 = tl.max(p1f, axis=0)
    e2 = tl.exp(p1f - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.bfloat16)

    # relu (no-op for positive softmax outputs, kept for exactness)
    out = tl.maximum(p2, tl.zeros_like(p2))

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            h, self.rms2_w, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
