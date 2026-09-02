import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 757
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)

    # x * 1.487 (opmath fp32, round back to bf16 like PyTorch elementwise)
    x = (x * 1.487).to(tl.bfloat16).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    # gelu (exact, erf) twice, rounding to bf16 after each op
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * n_cols + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.487
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, cols = x.shape[0] if x.dim() > 1 else 1, x.shape[-1]
        x2 = x.view(-1, cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](x2, y, cols, BLOCK=BLOCK, num_warps=num_warps)
        return y.view(x.shape)
