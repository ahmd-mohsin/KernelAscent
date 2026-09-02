import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 739
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16)
    x16 = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float16)

    # x = x * 1.4683 in fp16 (PyTorch computes in fp32 then rounds to fp16)
    xf = x16.to(tl.float32) * 1.4683
    x16 = xf.to(tl.float16)

    # RMSNorm in fp32
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * r).to(tl.float16)

    # multiply by rms weight (fp16 op, computed in fp32, rounded to fp16)
    w16 = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float16)
    z16 = (y16.to(tl.float32) * w16.to(tl.float32)).to(tl.float16)

    # softmax in fp32, output fp16
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.math.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_softmax_kernel[(m,)](
            h, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
