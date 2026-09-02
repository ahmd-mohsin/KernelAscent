import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 897
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_relu_softmax_rms2_kernel(
    x_ptr, w2_ptr, w3_ptr, out_ptr,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load (bf16 -> fp32, exact) ----
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- relu (exact in any precision) ----
    x = tl.maximum(x, 0.0)

    # ---- softmax in fp32 (matches torch's fp32 accumulation for bf16) ----
    xs = tl.where(mask, x, float('-inf'))
    m = tl.max(xs, axis=0)
    e = tl.exp(xs - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # cast to bf16 (softmax output dtype) then back to fp32 for the next op
    pf = p.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    sq = tl.where(mask, pf * pf, 0.0)
    ms1 = tl.sum(sq, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    y = (pf * r1).to(tl.bfloat16).to(tl.float32)

    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y * w2).to(tl.bfloat16).to(tl.float32)  # bf16 mul semantics (fp32 compute, bf16 round)

    # ---- RMSNorm 2 ----
    sq2 = tl.where(mask, z * z, 0.0)
    ms2 = tl.sum(sq2, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    y2 = (z * r2).to(tl.bfloat16).to(tl.float32)

    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y2 * w3).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_relu_softmax_rms2_kernel[(rows,)](
            x2d, self.rms2_w, self.rms3_w, out,
            N, x2d.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
