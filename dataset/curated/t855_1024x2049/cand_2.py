import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 855
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (x * r).to(tl.float16)  # cast to fp16 like reference
    w = tl.load(W + cols, mask=mask, other=0.0)
    z = (xn * w).to(tl.float16)  # fp16 multiply as in reference
    # exact GELU (erf), computed in fp32 (matches PyTorch opmath), cast to fp16
    zf = z.to(tl.float32)
    g = zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 tensor-core matmul
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y, n,
            x.stride(0), y.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
