import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 32
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_post_kernel(
    X, LN_G, LN_B, B4, RMS_W, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, matching PyTorch half layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g + b
    # round to fp16 like the reference (layer_norm output dtype is half)
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 accumulation, output rounded to fp16) ----
    y_sm = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_sm, axis=0)
    num = tl.exp(y_sm - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    y = num / denom
    y = y.to(tl.float16).to(tl.float32)

    # ---- GELU (exact, erf-based, fp32 math) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y16 = y.to(tl.float16)

    # ---- Add bias in fp16 (matches half + half tensor add) ----
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    y16 = y16 + b4

    # ---- RMSNorm: fp32 mean of squares, cast to fp16, multiply weight in fp16 ----
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    out = (yf * rrms).to(tl.float16) * w

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.b4, self.rms5_w, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
