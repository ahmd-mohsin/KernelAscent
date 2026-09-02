import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 428
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b1_ptr, b3_ptr, out_ptr, n_cols, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)

    # relu (fp16)
    x = tl.maximum(x, 0.0)
    # + b1, round to fp16 to match reference dtype semantics
    x = (x + b1).to(tl.float16)
    # exact gelu computed in fp32, cast back to fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.float16)
    # + b3, round to fp16
    x = (x + b3).to(tl.float16)

    # softmax in fp32 (matches PyTorch fp16 softmax which accumulates in fp32)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            x = F.gelu(x)
            x = x + self.b3
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(rows,)](
            x, self.b1, self.b3, out,
            cols, x.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
