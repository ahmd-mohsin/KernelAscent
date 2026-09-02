import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 38
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _double_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 accumulation, matching PyTorch half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(tl.where(mask, e1, 0.0), axis=0)
    y1 = e1 / s1
    # Round to fp16 (intermediate tensor dtype in the reference), then back to fp32
    y1h = y1.to(tl.float16)
    y2in = y1h.to(tl.float32)

    # Second softmax
    y2in = tl.where(mask, y2in, float('-inf'))
    m2 = tl.max(y2in, axis=0)
    e2 = tl.exp(y2in - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_softmax_kernel[(rows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
