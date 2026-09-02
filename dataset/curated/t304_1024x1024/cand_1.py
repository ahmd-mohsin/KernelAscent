import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 304
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact erf-based GELU (matches F.gelu default), computed in fp32 then
    # rounded back to bf16 to match the reference's per-op bf16 rounding
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g_bf16 = g.to(tl.bfloat16)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    z = g_bf16 + b  # bf16 addition, as in the reference

    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float("-inf"))
    row_max = tl.max(zf, axis=0)
    e = tl.exp(zf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            h = x @ self.W0
            h = F.gelu(h)
            h = h + self.b2
            return torch.softmax(h, dim=-1)

        # cuBLAS tensor-core matmul
        h = torch.matmul(x, self.W0)

        rows, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_bias_softmax_kernel[(rows,)](
            h, self.b2, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
