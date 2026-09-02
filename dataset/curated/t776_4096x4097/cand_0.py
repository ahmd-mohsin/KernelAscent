import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 776
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    X, B1, G, Bb, W, Out,
    D, stride_x, stride_o,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 math, fp16 result, matching torch.softmax on fp16) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = (e / s).to(tl.float16)

    # ---- bias add (fp16 arithmetic like PyTorch) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    y = p + b1

    # ---- layer norm (fp32 stats & affine, fp16 output) ----
    yf = y.to(tl.float32)
    n = D.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), 0) / n
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / n
    inv = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(Bb + offs, mask=mask, other=0.0).to(tl.float32)
    z = (diff * inv * g + bb).to(tl.float16)

    # ---- rms norm (fp32 math, cast fp16, then fp16 weight mult) ----
    zf = z.to(tl.float32)
    ms = tl.sum(tl.where(mask, zf * zf, 0.0), 0) / n
    r = (zf * (1.0 / tl.sqrt(ms + eps_rms))).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    out = r * w

    # ---- scalar multiply in fp16 (matches half_tensor * python_float) ----
    scale = tl.full((), 1.0801, tl.float16)
    out = out * scale

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(rows,)](
            x2, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            d, x2.stride(0), out.stride(0),
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        x = torch.softmax(x, dim=-1)
        x = x + self.b1
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        x = x * 1.0801
        return x
