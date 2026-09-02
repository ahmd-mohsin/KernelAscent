import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 68
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulate, fp16 round-trip like PyTorch) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    x_in = tl.where(mask, x, float('-inf'))
    m = tl.max(x_in, 0)
    e = tl.exp(x_in - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32 math, fp16 output * fp16 weight) ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    xh = (x * r).to(tl.float16) * w

    # ---- bias add (fp16) ----
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    xh = xh + b

    # ---- exact GELU (fp32 internal math, fp16 out) ----
    xf = xh.to(tl.float32)
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h, self.rms3_w, self.b4, y,
            N,
            h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
