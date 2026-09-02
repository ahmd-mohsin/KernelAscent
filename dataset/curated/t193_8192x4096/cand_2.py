import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 193
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _rms_softmax_kernel(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row (fp16) and upcast to fp32, matching _xf = x.float()
    xf = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS: mean of squares over last dim
    ms = tl.sum(xf * xf, axis=0) / N
    rr = tl.math.rsqrt(ms + 1e-6)

    # (xf * rsqrt).to(fp16) * w   (fp16 multiply, as in reference)
    y16 = (xf * rr).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    z16 = y16 * w

    # Softmax computed in fp32 (matches PyTorch half softmax accumulation)
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Elementwise scale in fp16 (same numerics as reference)
        x = x * 1.0122
        # cuBLAS GEMM (tensor cores)
        h = x @ self.W1
        h = h.contiguous()

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _rms_softmax_kernel[(m,)](
            h, self.rms2_w, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
