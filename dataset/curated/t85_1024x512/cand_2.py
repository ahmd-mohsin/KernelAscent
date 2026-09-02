import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 85
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    N, eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, as PyTorch does for fp16)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # cast to fp16 (matches layer_norm output dtype), then RMSNorm in fp32
    y16 = y.to(tl.float16)
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + eps_rms)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (yf * rinv).to(tl.float16)
    out = out * w  # fp16 multiply
    out = out * scale  # fp16 multiply by scalar

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        scale = torch.tensor(1.3885, dtype=torch.float16).item()
        _fused_ln_rms_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, y,
            x.stride(0), y.stride(0),
            N, 1e-5, 1e-6, scale,
            BLOCK=2048,
            num_warps=8,
        )
        return y
