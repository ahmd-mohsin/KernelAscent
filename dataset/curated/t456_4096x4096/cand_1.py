import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 456
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)
    # mask out-of-bounds for softmax
    x = tl.where(mask, x, float('-inf'))
    # numerically stable softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs via cuBLAS (tensor cores on A100)
        h = torch.mm(x, self.W0)
        h = torch.mm(h, self.W1)

        # Fused ReLU + softmax in a single Triton kernel (one row per program)
        M_rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _relu_softmax_kernel[(M_rows,)](
            h, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
