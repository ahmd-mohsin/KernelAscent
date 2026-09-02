import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 246
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (fp16 -> fp32) ----
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, matching torch fp16 softmax) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # torch stores softmax result back to fp16 before the RMS step
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    # ---- RMS norm (fp32), then cast to fp16, mul weight, add bias, relu ----
    ms = tl.sum(pf * pf, axis=0) / N
    rn = pf * tl.math.rsqrt(ms + 1e-6)
    rn16 = rn.to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    t = rn16 * w          # fp16 multiply (matches torch elementwise fp16)
    t = t + b             # fp16 add
    zero = tl.zeros(t.shape, dtype=tl.float16)
    t = tl.maximum(t, zero)  # relu in fp16

    # ---- softmax #2 (fp32 accumulation) ----
    tf = tl.where(mask, t.to(tl.float32), float('-inf'))
    m2 = tl.max(tf, axis=0)
    e2 = tl.exp(tf - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)

        _fused_post_kernel[(rows,)](
            h, self.rms2_w, self.b3, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
