import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 874
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _bias_scale_softmax_kernel(
    Y_ptr, B_ptr, Out_ptr,
    N, stride_y, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches reference), then scale
    t = (y + b).to(tl.float16)
    v = t.to(tl.float32) * SCALE
    v = tl.where(mask, v, float('-inf'))

    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS half matmul (tensor cores on A100)
        y = x @ self.W0
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _bias_scale_softmax_kernel[(Mrows,)](
            y, self.b1, out,
            N, y.stride(0), out.stride(0),
            SCALE=1.4874,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
