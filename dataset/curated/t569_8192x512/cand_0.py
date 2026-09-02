import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 569
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_bias_softmax_kernel(
    X, B, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X + row * N_COLS + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add in bf16 to match reference numerics (x + b0 in bf16)
    xb = (x.to(tl.bfloat16) + b.to(tl.bfloat16)).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    row_max = tl.max(xb, axis=0)
    e = tl.exp(xb - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * N_COLS + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.softmax(x, dim=-1)
            return torch.relu(x)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_bias_softmax_kernel[(n_rows,)](
            x, self.b0, y,
            N_COLS=n_cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        # softmax outputs are >= 0, so relu(relu(.)) is identity
        return y
