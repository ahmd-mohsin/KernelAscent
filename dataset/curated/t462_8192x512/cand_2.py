import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 462
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_rms_relu(X, W, Y, N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0)
    # relu in original dtype (bf16), then to fp32 (matches reference order)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)

    y = (xf * inv).to(Y.dtype.element_ty)  # round to bf16 (matches .to(x.dtype))
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = y * w                              # bf16 * bf16
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * D + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_relu_rms_relu[(m,)](
            x, self.rms2_w, y, m,
            D=d, BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
        return y
