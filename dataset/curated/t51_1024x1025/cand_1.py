import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 51
M, D, DT = 1024, 1025, torch.float16


@triton.jit
def _gelu_softmax_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 (matches PyTorch opmath for half)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round-trip through fp16 to match reference intermediate storage
    g = g.to(tl.float16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.dim() >= 1:
            orig_shape = x.shape
            n_cols = orig_shape[-1]
            x2 = x.contiguous().view(-1, n_cols)
            y = torch.empty_like(x2)
            n_rows = x2.shape[0]
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 4
            if BLOCK >= 2048:
                num_warps = 8
            if BLOCK >= 8192:
                num_warps = 16
            _gelu_softmax_kernel[(n_rows,)](
                x2, y, n_cols, x2.stride(0), y.stride(0),
                BLOCK=BLOCK, num_warps=num_warps,
            )
            return y.view(orig_shape)
        # fallback
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        x = torch.relu(x)
        x = torch.relu(x)
        return x
