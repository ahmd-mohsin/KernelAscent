import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 960
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_softmax_rms_gelu_softmax(
    X, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, output rounded to fp16 like PyTorch) ----
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, 0)
    y = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- RMSNorm (computed in fp32 on the fp16-rounded softmax output) ----
    msq = tl.sum(tl.where(mask, y * y, 0.0), 0) / D
    inv = 1.0 / tl.sqrt(msq + 1e-6)
    y = (y * inv).to(tl.float16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.float16).to(tl.float32)

    # ---- GELU (exact erf, fp32 opmath, rounded to fp16) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    y = tl.where(mask, y, float('-inf'))
    m2 = tl.max(y, 0)
    e2 = tl.exp(y - m2)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape[0], x.shape[-1]
        if x.dim() > 2:
            rows = x.numel() // d
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_gelu_softmax[(rows,)](
            x, self.rms1_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
