import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 493
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, round to bf16 like PyTorch output)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    x = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax 2
    x_in = tl.where(mask, x, float('-inf'))
    m2 = tl.max(x_in, axis=0)
    e2 = tl.exp(x_in - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    x = (e2 / s2).to(tl.bfloat16).to(tl.float32)

    # scale (fp32 opmath, round to bf16)
    x = (x * 1.4655).to(tl.bfloat16).to(tl.float32)

    # exact GELU (fp32 opmath)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            x = x * 1.4655
            return F.gelu(x)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, y, n, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
