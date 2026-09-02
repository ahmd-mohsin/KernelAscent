import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 137
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_gelu_softmax(X, W, Y, N, stride_x, stride_y, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean of squares in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xn * w).to(tl.float16)  # fp16 multiply as in reference

    # GELU (exact, computed in fp32 like PyTorch CUDA opmath, cast to fp16)
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # Softmax (fp32 accumulate, like PyTorch)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_gelu_softmax[(M_,)](
            x, self.rms1_w, y, N_, x.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
