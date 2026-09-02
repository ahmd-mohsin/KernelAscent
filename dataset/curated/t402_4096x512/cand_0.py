import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 402
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, G2, B2, W4, G5, B5, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), rounded to bf16 like reference
    h = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    h = h.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2 (fp32 stats, bf16 output)
    mean = tl.sum(h, axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g2 + b2).to(tl.bfloat16).to(tl.float32)

    # scale by 1.003 (fp32 math, bf16 round)
    y = (y * 1.003).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (fp32), then bf16 round, then bf16*bf16 weight mul (fp32 math, bf16 round)
    ms = tl.sum(y * y, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y * r).to(tl.bfloat16).to(tl.float32)
    z = (z * w4).to(tl.bfloat16).to(tl.float32)

    # LayerNorm 5
    mean2 = tl.sum(z, axis=0) / N
    d2 = tl.where(mask, z - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = tl.math.rsqrt(var2 + 1e-5)
    g5 = tl.load(G5 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (d2 * rstd2 * g5 + b5).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x * 1.003
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        # tensor-core matmul (cuBLAS)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N).contiguous()
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h2, self.ln2_g, self.ln2_b, self.rms4_w, self.ln5_g, self.ln5_b, out,
            N, h2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.reshape(orig_shape)
