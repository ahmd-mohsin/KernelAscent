import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 848
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X, Y, B,
    stride_xm, stride_ym,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = (e / s) * SCALE
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores)
        z = x @ self.W0
        # match reference: bias added in fp16 before softmax
        z = z + self.b1
        m, n = z.shape
        out = torch.empty_like(z)
        zero_bias = torch.zeros(1, device=z.device, dtype=z.dtype)
        BLOCK = triton.next_power_of_2(n)
        _softmax_scale_kernel[(m,)](
            z, out, zero_bias,
            z.stride(0), out.stride(0),
            N=n, SCALE=1.3952, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
