import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 321
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_rms_gelu_bias(
    x_ptr, w_ptr, b_ptr, out_ptr,
    n_cols, x_stride, out_stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    # ---- softmax (fp32 accumulate, like torch bf16 softmax) ----
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # cast to bf16 (softmax output dtype) then back up for rms, matching reference
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm ----
    ms = tl.sum(p * p, axis=0) / n_cols
    inv = tl.math.rsqrt(ms + EPS)
    r = (p * inv).to(tl.bfloat16)

    # multiply by weight (bf16 elementwise: fp32 compute, round to bf16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (r.to(tl.float32) * w).to(tl.bfloat16)

    # ---- exact GELU (erf-based, fp32 internal) ----
    g = y.to(tl.float32)
    g = g * 0.5 * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ---- bias add ----
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (g.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(out_ptr + row * out_stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xs = torch.softmax(x, dim=-1)
            _xf = xs.float()
            xs = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xs.dtype) * self.rms1_w
            xs = F.gelu(xs)
            return xs + self.b3

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_softmax_rms_gelu_bias[(n_rows,)](
            x2, self.rms1_w, self.b3, out,
            n_cols, x2.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
