import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 623
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_kernel(
    X_ptr, W1_ptr, G2_ptr, B2_ptr, W4_ptr, Out_ptr,
    D: tl.constexpr, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x16 = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # x = x * 1.3301  (opmath fp32, round to fp16)
    x = x16.to(tl.float32) * 1.3301
    x16 = x.to(tl.float16)
    x = x16.to(tl.float32)

    # RMSNorm 1 (fp32 accumulate, cast to fp16, multiply by w1 in fp16)
    xm = tl.where(mask, x, 0.0)
    ms = tl.sum(xm * xm, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0)
    xh = xh * w1
    x = xh.to(tl.float32)

    # LayerNorm (fp32 internal, eps 1e-5)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g2 = tl.load(G2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g2 + b2
    y16 = y.to(tl.float16)
    x = y16.to(tl.float32)

    # Softmax (fp32 internal)
    xneg = tl.where(mask, x, float("-inf"))
    mx = tl.max(xneg, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)
    x = p16.to(tl.float32)

    # RMSNorm 2
    xm = tl.where(mask, x, 0.0)
    ms2 = tl.sum(xm * xm, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    oh = (x * r2).to(tl.float16)
    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0)
    oh = oh * w4

    tl.store(Out_ptr + row * stride_o + offs, oh, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xx = x * 1.3301
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms1_w
            xx = F.layer_norm(xx, (xx.shape[-1],), self.ln2_g, self.ln2_b)
            xx = torch.softmax(xx, dim=-1)
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms4_w
            return xx

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)
