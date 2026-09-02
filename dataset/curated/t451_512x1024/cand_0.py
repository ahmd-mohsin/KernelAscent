import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 451
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_softmax(
    X, B, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X + row * N_COLS + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in bf16 (match reference rounding), then relu
    v = (x + b).to(tl.bfloat16)
    v = tl.maximum(v, 0.0)

    # softmax in fp32 (matches torch's internal fp32 accumulation)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * N_COLS + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_bias_relu_softmax[(n_rows,)](
            x, self.b0, y,
            N_COLS=n_cols,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
