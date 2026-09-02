import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 17
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax_ln(
    Y_ptr, B_ptr, W_ptr, G_ptr, Beta_ptr, OUT_ptr,
    N, stride_y, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- bias add (opmath fp32, cast back to bf16 like PyTorch elementwise) ----
    y = tl.load(Y_ptr + row * stride_y + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = (y + b).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm: fp32 stats, cast to bf16, then bf16*bf16 mul (fp32 opmath) ----
    ms = tl.sum(tl.where(mask, t * t, 0.0), axis=0) / N
    rinv = tl.math.rsqrt(ms + 1e-6)
    r = (t * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = (r * w).to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 accumulate, cast result to bf16) ----
    s_masked = tl.where(mask, s, float('-inf'))
    m = tl.max(s_masked, axis=0)
    e = tl.exp(s_masked - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.bfloat16).to(tl.float32)

    # ---- layer norm (fp32 math, biased variance, eps=1e-5) ----
    mean = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (d * inv * g + beta).to(tl.bfloat16)

    tl.store(OUT_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = x @ self.W0
            y = y + self.b1
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            y = torch.softmax(y, dim=-1)
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        y = torch.matmul(x, self.W0)  # cuBLAS bf16 GEMM (tensor cores)
        y = y.contiguous()
        orig_shape = y.shape
        N = orig_shape[-1]
        y2 = y.view(-1, N)
        rows = y2.shape[0]
        out = torch.empty_like(y2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_rms_softmax_ln[(rows,)](
            y2, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, out,
            N, y2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
