import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 703
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_post_kernel(
    X_ptr, W1_ptr, G_ptr, B_ptr, W5_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (fp32 math, round to fp16, then * w in opmath fp32) ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    x1 = (x * r).to(tl.float16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = (x1.to(tl.float32) * w1).to(tl.float16)

    # ---- LayerNorm (fp32 accum, biased variance, eps=1e-5) ----
    xf = x2.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * inv * g + b).to(tl.float16)

    # ---- ReLU then * 1.455 (opmath fp32, round back to fp16) ----
    y = tl.maximum(y, tl.zeros_like(y))
    y2 = (y.to(tl.float32) * 1.455).to(tl.float16)

    # ---- RMSNorm 2 (fp32 math, round to fp16, then * w in opmath fp32) ----
    yf = y2.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    y3 = (yf * r2).to(tl.float16)
    w5 = tl.load(W5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y3.to(tl.float32) * w5).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (fp16 tensor cores, fp32 accumulate)
        h = torch.matmul(x, self.W0)

        if not h.is_cuda:
            # CPU fallback: reference path
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms1_w
            h = F.layer_norm(h, (h.shape[-1],), self.ln2_g, self.ln2_b)
            h = torch.relu(h)
            h = h * 1.455
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms5_w
            return h

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_post_kernel[(rows,)](
            h2, self.rms1_w, self.ln2_g, self.ln2_b, self.rms5_w, out,
            N, h2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
