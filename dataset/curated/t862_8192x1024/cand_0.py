import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 862
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _ln_scale_gelu_kernel(
    X, G, B, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # mean / variance (biased) in fp32, matching PyTorch layer_norm internals
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    ln = (x - mean) * rstd * g + b
    # cast to fp16 (LN output dtype), then scale in fp16 like reference
    ln_h = ln.to(tl.float16)
    scale_h = tl.full((1,), SCALE, dtype=tl.float32).to(tl.float16)
    y_h = ln_h * scale_h

    # exact GELU (erf) computed in fp32, cast back to fp16
    t = y_h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = t * 0.5 * (1.0 + tl.math.erf(t * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_scale_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, y,
            N, h.stride(0), y.stride(0),
            SCALE=1.4752,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
