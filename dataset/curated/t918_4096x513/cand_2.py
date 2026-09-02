import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 918
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x,
    EPS: tl.constexpr,
    S1: tl.constexpr,   # 1.1963
    S2: tl.constexpr,   # 1.042
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)                   # fp16
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)                   # fp16

    # x = x * 1.1963  (compute fp32, round to fp16)
    h = (x.to(tl.float32) * S1).to(tl.float16)
    # x = x + b2      (fp32 opmath, round to fp16)
    h = (h.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # RMSNorm in fp32
    xf = h.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    h = (xf * inv).to(tl.float16)

    # * rms3_w (fp32 opmath, round to fp16)
    h = (h.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # exact GELU in fp32, round to fp16
    g = h.to(tl.float32)
    g = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    h = g.to(tl.float16)

    # * 1.042 (fp32 opmath, round to fp16)
    out = (h.to(tl.float32) * S2).to(tl.float16)

    tl.store(Out_ptr + row * stride_x + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            y, self.b2, self.rms3_w, out,
            N, y.stride(0),
            EPS=1e-6, S1=1.1963, S2=1.042,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
