import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 207
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # x + b1  (fp32 math, round to fp16 like PyTorch half add)
    t = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    # * 1.0905 (fp32 math, round to fp16)
    t = (t.to(tl.float32) * SCALE).to(tl.float16)

    # exact GELU in fp32 (PyTorch opmath), round to fp16
    tf = t.to(tl.float32)
    g = (tf * 0.5 * (1.0 + tl.math.erf(tf * 0.7071067811865476))).to(tl.float16)

    # RMSNorm: cast fp16 -> fp32, mean of squares over row
    gf = g.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    ms = tl.sum(gf * gf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    # (gf * r) -> fp16, then multiply by weight (fp32 opmath, round fp16)
    y = (gf * r).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            h, self.b1, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-6, SCALE=1.0905,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
