import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 520
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_rms_ln(
    X, W0, W3, G, B, Y,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 (compute in fp32, round to bf16, then bf16 weight mul) ----
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    a = (x * r).to(tl.bfloat16).to(tl.float32)
    a = (a * w0).to(tl.bfloat16).to(tl.float32)

    # ---- scalar scale (bf16 rounding, matching bf16 tensor * python float) ----
    a = (a * 1.1232).to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 internal, bf16 output like torch.softmax on bf16) ----
    mx = tl.max(tl.where(mask, a, float('-inf')), axis=0)
    e = tl.exp(a - mx)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, axis=0)
    s = s.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 ----
    ms2 = tl.sum(s * s, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    d = (s * r2).to(tl.bfloat16).to(tl.float32)
    d = (d * w3).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 internal, eps=1e-5, bf16 output) ----
    mean = tl.sum(tl.where(mask, d, 0.0), axis=0) / D
    dc = tl.where(mask, d - mean, 0.0)
    var = tl.sum(dc * dc, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    y = dc * inv * g + b

    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_rms_softmax_rms_ln[(rows,)](
            x2, self.rms0_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x * 1.1232
        x = torch.softmax(x, dim=-1)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
