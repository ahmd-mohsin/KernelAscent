import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 501
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_ln_ln_relu_rms_kernel(
    X, Y, G0, B0, G1, B1, W,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, output rounded to bf16) ----
    mean0 = tl.sum(x, 0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, 0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + 1e-5)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d0 * rstd0 * g0 + b0
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, output rounded to bf16) ----
    mean1 = tl.sum(tl.where(mask, x, 0.0), 0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d1 * rstd1 * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU (exact in bf16 -> fp32) ----
    x = tl.maximum(x, 0.0)

    # ---- RMSNorm (fp32 math, cast to bf16, bf16 multiply by weight) ----
    ms = tl.sum(x * x, 0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (x * rrms).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.bfloat16)
    y = xb * w

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_relu_rms_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.rms3_w,
            N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
