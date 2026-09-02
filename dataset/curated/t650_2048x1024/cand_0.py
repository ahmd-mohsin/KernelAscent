import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 650
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_scale_bias_softmax(
    X, B2, B3, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS

    x = tl.load(X + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # replicate per-op bf16 rounding of the reference elementwise chain
    x = (x * 1.1314).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.4149).to(tl.bfloat16).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)
    x = (x + b3).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.2954).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax internal accumulation)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * N_COLS + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.1314
            x = x * 1.4149
            x = x + self.b2
            x = x + self.b3
            x = x * 1.2954
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_scale_bias_softmax[(n_rows,)](
            x2d, self.b2, self.b3, y,
            N_COLS=n_cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
