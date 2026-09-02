import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 810
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # scale (round to fp16 like PyTorch elementwise op on half tensors)
    x = x * 1.2221
    x = x.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf form)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # gelu #2
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # softmax over the row (fp32 accumulation, like PyTorch's half softmax)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        m, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        _fused_act_softmax_kernel[(m,)](
            h, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=1,
        )
        return out
