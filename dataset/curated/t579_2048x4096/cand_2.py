import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 579
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_softmax_gelu(X, W, Y, N, stride_x, stride_y, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, result cast to fp16 like reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    xn = (x * inv).to(tl.float16).to(tl.float32) * w
    xn = xn.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation as PyTorch does for half inputs)
    xn_m = tl.where(mask, xn, float('-inf'))
    mx = tl.max(xn_m, axis=0)
    e = tl.exp(xn_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # ReLU (identity on softmax output but kept) + exact GELU
    r = tl.maximum(sm, 0.0)
    g = r * 0.5 * (1.0 + tl.math.erf(r * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_gelu[(Mrows,)](
            x, self.rms1_w, y, N, x.stride(0), y.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=4,
        )
        return y
