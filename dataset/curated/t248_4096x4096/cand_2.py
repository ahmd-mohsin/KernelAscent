import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 248
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _bias_gelu_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (compute fp32, round to bf16 to match eager op semantics)
    v = x + b
    v = v.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * v * (1 + erf(v / sqrt(2)))
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation)
    g = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * N + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _bias_gelu_softmax_kernel[(rows,)](
            h, self.b1, y,
            N,
            BLOCK=BLOCK,
            num_warps=16,
            num_stages=1,
        )
        return y
