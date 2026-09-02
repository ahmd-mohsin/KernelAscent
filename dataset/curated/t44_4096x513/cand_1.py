import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 44
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_bias_softmax_gelu_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N,  # number of columns (512)
    stride_xm,
    stride_om,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add (opmath float, round to half like PyTorch)
    x = (x + b).to(tl.float16).to(tl.float32)

    # softmax in fp32, output rounded to half
    x_masked = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_masked, axis=0)
    e = tl.exp(x_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = (e / denom).to(tl.float16).to(tl.float32)

    # exact (erf) gelu in fp32, output rounded to half
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = (s * 0.5 * (1.0 + tl.math.erf(s * INV_SQRT2))).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    g2 = tl.where(mask, g * g, 0.0)
    ms = tl.sum(g2, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    y = (g * r).to(tl.float16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulate
        m, n = x.shape
        out = torch.empty_like(x)
        grid = (m,)
        _fused_bias_softmax_gelu_rms_kernel[grid](
            x, self.b1, self.rms4_w, out,
            n,
            x.stride(0),
            out.stride(0),
            EPS=1e-6,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
