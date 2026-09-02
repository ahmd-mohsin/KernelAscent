import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 416
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, OUT,
    LN0_G, LN0_B, RMS1_W, LN2_G, LN2_B, RMS3_W,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    g0 = tl.load(LN0_G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0_B + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(RMS1_W + cols, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(LN2_G + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(LN2_B + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(RMS3_W + cols, mask=mask, other=0.0).to(tl.float32)

    nf = N.to(tl.float32)

    # --- LayerNorm 0 (fp32 math, output rounded to bf16) ---
    mean = tl.sum(x, axis=0) / nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    x = (d * rstd) * g0 + b0
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 1 ---
    ms = tl.sum(x * x, axis=0) / nf
    x = x * (1.0 / tl.sqrt(ms + RMS_EPS))
    x = x.to(tl.bfloat16).to(tl.float32)
    x = x * w1
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm 2 ---
    mean = tl.sum(x, axis=0) / nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    x = (d * rstd) * g2 + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm 3 ---
    ms = tl.sum(x * x, axis=0) / nf
    x = x * (1.0 / tl.sqrt(ms + RMS_EPS))
    x = x.to(tl.bfloat16).to(tl.float32)
    x = x * w3

    tl.store(OUT + row * stride_o + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = y.float(); y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms1_w
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            _xf = y.float(); y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_norms_kernel[(rows,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b, self.rms3_w,
            N, x2.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
