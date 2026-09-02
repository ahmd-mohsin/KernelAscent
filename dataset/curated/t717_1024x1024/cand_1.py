import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 717
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    LN0G, LN0B, LN1G, LN1B, RMS2W, LN3G, LN3B, RMS4W,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    ln_eps: tl.constexpr = 1e-5
    rms_eps: tl.constexpr = 1e-6
    inv_n = 1.0 / N

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g = tl.load(LN0G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN0B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g = tl.load(LN1G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN1B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 (mul by weight in fp16, matching reference) ----
    ms = tl.sum(x * x, axis=0) * inv_n
    rs = 1.0 / tl.sqrt(ms + rms_eps)
    w = tl.load(RMS2W + cols, mask=mask, other=0.0)  # fp16
    xh = (x * rs).to(tl.float16) * w
    x = xh.to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 4 ----
    ms = tl.sum(x * x, axis=0) * inv_n
    rs = 1.0 / tl.sqrt(ms + rms_eps)
    w = tl.load(RMS4W + cols, mask=mask, other=0.0)  # fp16
    out = (x * rs).to(tl.float16) * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._forward_ref(x)
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_norms_kernel[(Mrows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            self.rms2_w, self.ln3_g, self.ln3_b, self.rms4_w,
            x2.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
        x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
