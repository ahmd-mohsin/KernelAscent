import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 757
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # x * 1.487  (bf16 op: compute in fp32, round to bf16)
    x = (x * 1.487).to(tl.bfloat16).to(tl.float32)
    # relu (exact in bf16)
    x = tl.maximum(x, 0.0)
    # gelu (exact, erf) with bf16 rounding after each op
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's bf16 softmax which uses fp32 accumulation)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            y = x * 1.487
            y = torch.relu(y)
            y = F.gelu(y)
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        if BLOCK > 32768:
            z = x * 1.487
            z = torch.relu(z)
            z = F.gelu(z)
            z = F.gelu(z)
            return torch.softmax(z, dim=-1)

        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_kernel[(n_rows,)](
            x2d, y, n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
