import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 800
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    X, B, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # bf16

    # bias add in bf16 (matches x + self.b1 in bf16)
    xb = x + b

    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)

    y_bf16 = (xf * rs).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # bf16
    out = y_bf16 * w

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        if not h.is_cuda:
            hb = h + self.b1
            _xf = hb.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(hb.dtype) * self.rms2_w

        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _bias_rmsnorm_kernel[(m,)](
            h, self.b1, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
