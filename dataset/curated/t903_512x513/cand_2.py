import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 903
M, D, DT = 512, 513, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf), round to fp16 to match reference intermediate precision
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # gelu #2
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # softmax (fp32 accumulation)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS/tensor-core GEMM
        if not h.is_cuda:
            h = F.gelu(h)
            h = F.gelu(h)
            h = torch.relu(h)
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_act_softmax_kernel[(m,)](
            h, out, n, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
