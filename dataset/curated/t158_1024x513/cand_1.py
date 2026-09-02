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
    n_cols,
    stride_x, stride_out,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)

    # x = x * 1.0026 (compute fp32, round to bf16 like PyTorch elementwise)
    xf = x.to(tl.float32) * 1.0026
    x = xf.to(tl.bfloat16)

    # x = x + b1
    xf = x.to(tl.float32) + b1.to(tl.float32)
    x = xf.to(tl.bfloat16)

    # softmax in fp32 accumulation (matches PyTorch bf16 softmax)
    xf = x.to(tl.float32)
    xf_masked = tl.where(mask, xf, float("-inf"))
    row_max = tl.max(xf_masked, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # x = x + b3
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)
    yf = sm.to(tl.float32) + b3.to(tl.float32)
    y = yf.to(tl.bfloat16)

    # x = x * 1.0435
    yf = y.to(tl.float32) * 1.0435
    y = yf.to(tl.bfloat16)

    tl.store(out_ptr + row * stride_out + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.0026
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            y = y + self.b3
            y = y * 1.0435
            return y

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_kernel[(n_rows,)](
            x2d, self.b1, self.b3, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
