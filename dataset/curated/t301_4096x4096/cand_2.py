import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 301
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_scale_softmax_ln_relu(
    X, G, B, Y,
    stride_x, stride_y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # scale (match PyTorch half elementwise: fp32 compute, round to fp16)
    x = (x * scale).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, output rounded to fp16 like PyTorch)
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # layer norm (fp32 stats)
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_scale_softmax_ln_relu[(Mrows,)](
            h, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            N, 1e-5, 1.0477,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
