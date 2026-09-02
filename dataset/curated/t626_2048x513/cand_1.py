import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 626
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, g_ptr, b_ptr, out_ptr,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0029  (fp16 output rounding, fp32 opmath)
    y = x * SCALE
    yh = y.to(tl.float16)
    yf = yh.to(tl.float32)

    # RMSNorm in fp32, cast to fp16
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + RMS_EPS)
    a = (yf * inv).to(tl.float16)

    # multiply by rms weight (fp16 tensors, fp32 opmath, fp16 result)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = (a.to(tl.float32) * w).to(tl.float16)
    bf = b.to(tl.float32)

    # softmax (fp32 accumulation, fp16 output)
    mmax = tl.max(tl.where(mask, bf, float('-inf')), axis=0)
    e = tl.exp(bf - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)
    smf = sm.to(tl.float32)

    # layer norm (fp32 stats, fp16 output)
    mean = tl.sum(tl.where(mask, smf, 0.0), axis=0) / N
    diff = tl.where(mask, smf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    invstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((smf - mean) * invstd * g + bb).to(tl.float16)

    tl.store(out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._ref_forward(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(rows,)](
            x2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            N, x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5, SCALE=1.0029,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.reshape(orig_shape)

    def _ref_forward(self, x):
        x = x * 1.0029
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = torch.softmax(x, dim=-1)
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        return x
