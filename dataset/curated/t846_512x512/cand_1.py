import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 846
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_softmax_kernel(
    x_ptr, b1_ptr, b2_ptr, out_ptr,
    n_cols,
    stride_row,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # Match PyTorch half elementwise semantics: compute in fp32, round to fp16 after each op
    v = (x.to(tl.float32) * scale).to(tl.float16)
    v = (v.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    v = (v.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)

    # Softmax in fp32 (matches PyTorch's float accumulation for half)
    vf = tl.where(mask, v.to(tl.float32), float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.exp(vf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    result = (e / denom).to(tl.float16)

    tl.store(out_ptr + row * stride_row + cols, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2d = x.view(-1, n_cols)
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_kernel[(x2d.shape[0],)](
            x2d, self.b1, self.b2, out,
            n_cols, x2d.stride(0),
            1.1929,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(x)
