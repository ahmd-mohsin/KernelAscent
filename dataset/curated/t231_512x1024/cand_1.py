import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 231
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_rms_softmax_relu(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean of squares over the row, computed in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # match: (_xf * rsqrt).to(bf16) * w  (bf16 elementwise mul)
    xn = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax on bf16 input, fp32 accumulation, output bf16
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    # relu (no-op numerically after softmax, kept for equivalence)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_relu[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
