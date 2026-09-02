import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 6
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _ln_rms_kernel(
    X, G, B, W, OUT,
    stride_x, stride_o,
    N,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, bf16 output like PyTorch)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * inv_std * g + b
    y_bf16 = y.to(tl.bfloat16)

    # RMSNorm on the bf16 layernorm output (cast to fp32 as in reference)
    yf = y_bf16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rms = 1.0 / tl.sqrt(ms + RMS_EPS)

    t = (yf * rms).to(tl.bfloat16)  # rounding to bf16 as in reference `.to(x.dtype)`

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM
        h = h.contiguous()
        M_, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_kernel[(M_,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            h.stride(0), out.stride(0),
            N,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
