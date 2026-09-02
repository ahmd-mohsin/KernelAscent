import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 925
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _scale_bias_gelu_kernel(X, B, Y, n_cols, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // tl.cdiv(n_cols, BLOCK)
    cb = pid % tl.cdiv(n_cols, BLOCK)
    cols = cb * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    off = row * n_cols + cols

    x = tl.load(X + off, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # match reference rounding: bf16 mul, bf16 add, gelu in fp32 opmath
    t = (x.to(tl.float32) * 1.045).to(tl.bfloat16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    tf = t.to(tl.float32)
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * 0.7071067811865476))
    tl.store(Y + off, g.to(tl.bfloat16), mask=mask)


@triton.jit
def _softmax_kernel(X, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * n_cols + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # (M, 2048) via cuBLAS
        h = h.contiguous()
        m, n = h.shape
        act = torch.empty_like(h)
        BLOCK = 1024
        grid = (m * triton.cdiv(n, BLOCK),)
        _scale_bias_gelu_kernel[grid](h, self.b2, act, n, BLOCK=BLOCK, num_warps=4)

        z = act @ self.W4  # (M, 1024)
        z = z.contiguous()
        out = torch.empty_like(z)
        n2 = z.shape[1]
        _softmax_kernel[(z.shape[0],)](z, out, n2, BLOCK=triton.next_power_of_2(n2), num_warps=8)
        return out
