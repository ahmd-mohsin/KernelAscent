import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 197
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _gelu_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, matching PyTorch's half->float opmath then round to half
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # RMS over fp32 view of the fp16 gelu output
    ms = tl.sum(gf * gf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    y16 = (gf * inv).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = y16 * w

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu_rmsnorm_kernel[(m,)](
            h, self.rms2_w, y,
            h.stride(0), y.stride(0),
            n, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
