import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 660
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _softmax_gelu_kernel(
    x_ptr, out_ptr,
    n_cols,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # cast to fp16 and back to match softmax-output dtype before gelu
    sm = sm.to(tl.float16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * sm * (1.0 + tl.math.erf(sm * inv_sqrt2))

    tl.store(out_ptr + row * stride_o + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4
        _softmax_gelu_kernel[(m,)](
            x, out, n,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out @ self.W2
