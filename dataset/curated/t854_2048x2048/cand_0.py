import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 854
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)

    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # round softmax to bf16 (matches torch.softmax output dtype),
    # then add bias with float32 opmath (matches PyTorch elementwise add)
    sm_bf16 = sm.to(tl.bfloat16)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    out = (sm_bf16.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x, dim=-1) + self.b1

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _softmax_bias_kernel[(n_rows,)](
            x2d, self.b1, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
