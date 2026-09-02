import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 879
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_relu_rms_ln_kernel(
    X, W_rms, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # RMSNorm (fp32 stats), round to fp16, multiply by weight (fp32 mul, round to fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    h = (xf * inv).to(tl.float16)

    w = tl.load(W_rms + cols, mask=mask, other=0.0)
    h = (h.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # LayerNorm in fp32 (matches PyTorch opmath for half)
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    diff = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (hf - mean) * rstd * g + b

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms_ln_kernel[(Mrows,)](
            h, self.rms2_w, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
