import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 845
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_act_softmax_kernel(X_ptr, Y_ptr, N, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X_ptr + row * stride_row + offs
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf) computed in fp32, rounded back to fp16 to match
    # PyTorch's per-op half-precision storage
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # second gelu
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch's half softmax)
    x = tl.where(mask, x, float('-inf'))
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
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 tensor-core GEMM
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_act_softmax_kernel[(rows,)](
            h, out, N, h.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
