import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 985
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, N_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(x_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x * 1.1389 (fp32 compute, round to bf16 like PyTorch CUDA elementwise)
    y = (x * 1.1389).to(tl.bfloat16).to(tl.float32)
    # x + b1
    y = (y + b).to(tl.bfloat16).to(tl.float32)
    # x * 1.0458
    y = (y * 1.0458).to(tl.bfloat16).to(tl.float32)
    # relu
    y = tl.maximum(y, 0.0)
    # gelu (erf-based, fp32 compute, round to bf16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = (y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(out_ptr + row * N_COLS + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2 = x.view(-1, cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2, self.b1, out, cols, BLOCK=BLOCK, num_warps=num_warps
        )
        return out.view(x.shape)
