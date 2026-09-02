import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 650
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b2_ptr, b3_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)

    # Replicate eager-mode bf16 rounding per op
    x = (x.to(tl.float32) * 1.1314).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.4149).to(tl.bfloat16)
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.2954).to(tl.bfloat16)

    # Softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_om + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.1314
            x = x * 1.4149
            x = x + self.b2
            x = x + self.b3
            x = x * 1.2954
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_kernel[(m,)](
            x, self.b2, self.b3, out,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
