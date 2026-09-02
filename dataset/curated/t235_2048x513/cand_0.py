import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 235
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _gelu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = tl.where(mask, g, float('-inf'))

    # softmax
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = y + self.b1
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        # fused matmul + bias via cuBLAS (tensor cores on A100)
        h = torch.addmm(self.b1, x, self.W0)

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _gelu_softmax_kernel[(Mrows,)](
            h, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
