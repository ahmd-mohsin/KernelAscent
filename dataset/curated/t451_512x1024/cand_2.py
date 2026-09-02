import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 451
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # bias add + relu in bf16 (matches reference elementwise ops)
    v = x + b
    zero = tl.zeros_like(v)
    v = tl.maximum(v, zero)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.exp(vf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_relu_softmax[(x2.shape[0],)](
            x2, self.b0, out,
            n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(x)
