import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 464
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_relu_softmax_gelu_kernel(
    X, Y, N_COLS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    x = tl.load(X + row * N_COLS + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # relu
    x = tl.where(x > 0.0, x, 0.0)
    x = tl.where(mask, x, float('-inf'))
    # softmax (fp32 accumulation, matching PyTorch's half softmax behavior)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to fp16 (softmax output dtype in reference), then gelu in fp32
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)
    # exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * pf * (1.0 + tl.math.erf(pf * INV_SQRT2))
    tl.store(Y + row * N_COLS + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)
        x = x.contiguous()
        n_rows, n_cols = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(-1, n_cols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_relu_softmax_gelu_kernel[(x2.shape[0],)](
            x2, y, n_cols, BLOCK, num_warps=num_warps
        )
        return y.view(x.shape)
