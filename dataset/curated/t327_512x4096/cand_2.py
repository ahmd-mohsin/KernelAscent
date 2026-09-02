import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 327
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_bias_scale_bias_softmax(
    X, B1, B3, OUT,
    stride_xm, stride_om,
    N, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)

    # replicate fp16 elementwise semantics (round at each step)
    scale = tl.full((), SCALE, dtype=tl.float16)
    v = (x + b1).to(tl.float16)
    v = (v * scale).to(tl.float16)
    v = (v + b3).to(tl.float16)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    row_max = tl.max(vf, axis=0)
    e = tl.exp(vf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS tensor-core GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_scale_bias_softmax[(m,)](
            y, self.b1, self.b3, out,
            y.stride(0), out.stride(0),
            n, 1.1132,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
