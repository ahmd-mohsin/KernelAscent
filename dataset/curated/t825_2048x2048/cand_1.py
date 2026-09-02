import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 825
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    X, B0, G1, Bt1, W2, Y,
    stride_xm, stride_ym,
    N, LN_EPS, RMS_EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0 in fp16 (matches PyTorch fp16 add), then upcast for LN stats
    xh = (x + b0).to(tl.float16)
    xf = xh.to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    bt1 = tl.load(Bt1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g1 + bt1

    # round to fp16 as layer_norm output, then RMSNorm in fp32
    yh = y.to(tl.float16)
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)
    zh = (yf * rrms).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    out = zh * w2  # fp16 multiply

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_rms_kernel[(Mrows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-5, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
