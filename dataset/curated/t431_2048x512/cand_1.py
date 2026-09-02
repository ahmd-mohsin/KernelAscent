import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 431
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(
    X, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X + row * N_COLS + cols, mask=mask, other=0.0)

    # scale (float opmath, round to fp16 like PyTorch)
    x32 = x.to(tl.float32) * 1.2649
    xh = x32.to(tl.float16)

    # exact GELU (erf variant) with float opmath, round to fp16
    v = xh.to(tl.float32)
    g = v * 0.5 * (1.0 + tl.math.erf(v * 0.7071067811865476))
    gh = g.to(tl.float16)

    # softmax in fp32
    s = gh.to(tl.float32)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * N_COLS + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = x * 1.2649
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, cols = x.shape[0], x.shape[-1]
        x2 = x.view(-1, cols)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_gelu_softmax[(rows,)](
            x2, y, cols, BLOCK=BLOCK, num_warps=num_warps
        )
        return y.view_as(x)
