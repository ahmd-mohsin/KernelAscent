import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, W2, Y,
    n_cols,
    stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (fp32, cast to fp16, then weight mul in fp32 opmath -> fp16)
    ms = tl.sum(x * x, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x1 = (x * r).to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (x1.to(tl.float32) * w0).to(tl.float16)

    # Softmax (fp32 accumulation, output fp16)
    s = x2.to(tl.float32)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.float16)

    # RMSNorm 2
    pf = p.to(tl.float32)
    ms2 = tl.sum(pf * pf, axis=0) / n_cols
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    y = (pf * r2).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w2).to(tl.float16)

    # scalar muls (fp32 opmath, round to fp16 after each, matching eager)
    y = (y.to(tl.float32) * S1).to(tl.float16)
    y = (y.to(tl.float32) * S2).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.dim() == 2
        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.rms0_w, self.rms2_w, y,
            n_cols,
            x.stride(0), y.stride(0),
            1.3308, 1.3952,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
