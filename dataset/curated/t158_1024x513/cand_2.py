import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 158
M, D, DT = 1024, 513, torch.bfloat16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b1_ptr, b3_ptr, out_ptr,
    n_cols, stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)

    # x = x * 1.0026 (bf16 rounding after op)
    t = (x.to(tl.float32) * 1.0026).to(tl.bfloat16)
    # x = x + b1 (bf16 rounding after op)
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (as PyTorch does for bf16), output rounded to bf16
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # x = x + b3 (bf16 rounding)
    r = (sm.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    # x = x * 1.0435 (bf16 rounding)
    r = (r.to(tl.float32) * 1.0435).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_om + cols, r, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.0026
            x = x + self.b1
            x = torch.softmax(x, dim=-1)
            x = x + self.b3
            x = x * 1.0435
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_kernel[(m,)](
            x2, self.b1, self.b3, out,
            n, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
