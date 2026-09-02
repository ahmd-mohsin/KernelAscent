import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 874
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    x = x * scale
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fused matmul + bias via tensor-core addmm
        h = torch.addmm(self.b1, x, self.W0)

        # Fused scale + softmax in a single Triton kernel (one pass over memory)
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scale_softmax_kernel[(rows,)](
            h, out,
            N, h.stride(0), out.stride(0),
            1.4874,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
