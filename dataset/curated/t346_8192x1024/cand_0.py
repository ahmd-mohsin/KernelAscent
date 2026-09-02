import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 346
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 like PyTorch's opmath,
    # then rounded to fp16 to match the fp16 intermediate tensor.
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))

    # Softmax in fp32 accumulation (matches PyTorch fp16 softmax), round to fp16
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # ReLU (softmax output is non-negative, but keep for exactness)
    r = tl.maximum(sm, 0.0)

    # Bias add in fp32 opmath, round to fp16 (correctly-rounded fp16 add)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (r + b).to(tl.float16)

    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            return y + self.b3
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(n_rows,)](
            x, self.b3, out, n_cols, x.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
