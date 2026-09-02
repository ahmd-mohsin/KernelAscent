import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 241
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, RMS_W, LN_G, LN_B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast back to fp16, multiply by fp16 weight)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)  # fp16
    y_h = (xf * rrms).to(tl.float16) * w  # fp16 arithmetic
    yf = y_h.to(tl.float32)

    # Softmax (fp32 accumulation, fp16 output)
    yf_m = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf_m, axis=0)
    e = tl.exp(yf_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm_h = (e / s).to(tl.float16)  # fp16 softmax result
    smf = sm_h.to(tl.float32)

    # LayerNorm (stats in fp32)
    mean = tl.sum(tl.where(mask, smf, 0.0), axis=0) / N
    d = tl.where(mask, smf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((smf - mean) * rstd * g + b).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = torch.softmax(y, dim=-1)
            return F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_softmax_ln_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            x2.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
