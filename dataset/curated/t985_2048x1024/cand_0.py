import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 985
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, n_cols, stride_x, stride_y,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # elementwise chain, rounding to bf16 after each op to match PyTorch semantics
    x = (x * 1.1389).to(tl.bfloat16).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.0458).to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)
    # exact (erf-based) GELU, computed in fp32 like PyTorch opmath
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax over the row in fp32 accumulation
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.1389
            y = y + self.b1
            y = y * 1.0458
            y = torch.relu(y)
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, cols = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, self.b1, out, cols, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
