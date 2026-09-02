import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 98
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W2_ptr, B4_ptr, W5_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # fp16

    # x = x * 1.1988  (opmath fp32, rounded back to fp16)
    xf = x.to(tl.float32) * 1.1988
    x16 = xf.to(tl.float16)

    # RMSNorm #1
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    n16 = (xf * r).to(tl.float16)

    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)  # fp16
    v = (n16.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # relu (exact in fp16)
    v = tl.maximum(v, 0.0)

    # + b4 (fp32 add of fp16 values, rounded -> identical to correctly-rounded fp16 add)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0)
    v = (v.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # RMSNorm #2
    vf = v.to(tl.float32)
    ms2 = tl.sum(vf * vf, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    n2 = (vf * r2).to(tl.float16)

    w5 = tl.load(W5_ptr + cols, mask=mask, other=0.0)
    out = (n2.to(tl.float32) * w5.to(tl.float32)).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 matmul (same as reference)
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_post_kernel[(m,)](
            x, self.rms2_w, self.b4, self.rms5_w, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
