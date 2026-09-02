import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 578
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b0_ptr, b3_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)

    # match reference dtype semantics: (x + b0) in bf16, then * scale in bf16
    xb = (x + b0).to(tl.bfloat16)
    xs = (xb * SCALE).to(tl.bfloat16)

    v = tl.where(mask, xs.to(tl.float32), float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)
    out = sm + b3

    tl.store(out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax((x + self.b0) * 1.4701, dim=-1)
            return y + self.b3

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_softmax_kernel[(x2.shape[0],)](
            x2, self.b0, self.b3, out,
            n_cols,
            x2.stride(0), out.stride(0),
            SCALE=1.4701,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(x.shape)
