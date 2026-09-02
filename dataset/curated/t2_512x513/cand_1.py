import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 2
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_post_kernel(
    A_ptr, W_ptr, Out_ptr,
    stride_a, stride_o,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # ---- load matmul output row (fp16 -> fp32) ----
    a = tl.load(A_ptr + row * stride_a + offs).to(tl.float32)

    # ---- softmax #1 (fp32 math, round to fp16 like PyTorch) ----
    m1 = tl.max(a, axis=0)
    e1 = tl.exp(a - m1)
    s1 = tl.sum(e1, axis=0)
    x = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf-based, fp32 opmath, round to fp16) ----
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32 reduction), scale in fp16 like PyTorch ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16)
    w = tl.load(W_ptr + offs)
    y = xn * w  # fp16 multiply, matching torch half*half elementwise
    yf = y.to(tl.float32)

    # ---- softmax #2 ----
    m2 = tl.max(yf, axis=0)
    e2 = tl.exp(yf - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.softmax(x, dim=-1)

        # cuBLAS matmul (fp16 tensor cores, fp32 accumulate) - same as reference
        a = torch.matmul(x, self.W0)
        a = a.contiguous()
        rows, cols = a.shape  # cols == 1024

        out = torch.empty_like(a)
        _fused_post_kernel[(rows,)](
            a, self.rms3_w, out,
            a.stride(0), out.stride(0),
            cols,
            BLOCK=1024,
            num_warps=8,
        )
        return out
