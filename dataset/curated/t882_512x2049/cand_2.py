import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 882
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    x_ptr, w_ptr, out_ptr,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # --- GELU (exact, erf-based, computed in fp32 like PyTorch, cast back to bf16) ---
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # --- Softmax #1 (fp32 accumulation, bf16 output like PyTorch) ---
    g_masked = tl.where(mask, g, float('-inf'))
    m1 = tl.max(g_masked, axis=0)
    e1 = tl.math.exp(g_masked - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    sm1 = (e1 / s1).to(tl.bfloat16)

    # --- RMSNorm: _xf = sm1.float(); fp32 mean of squares ---
    xf = sm1.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    r = (xf * inv).to(tl.bfloat16)

    # --- multiply by weight (fp32 opmath, bf16 result) ---
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (r.to(tl.float32) * w).to(tl.bfloat16)

    # --- ReLU (on bf16) ---
    y = tl.maximum(y, 0.0)

    # --- Softmax #2 (fp32 accumulation, bf16 output) ---
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m2 = tl.max(yf, axis=0)
    e2 = tl.math.exp(yf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        w = self.rms2_w
        if w.device != x.device:
            w = w.to(x.device)

        _fused_row_kernel[(rows,)](
            x2, w, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
