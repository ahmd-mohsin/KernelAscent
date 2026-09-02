import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 73
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_post_kernel(
    X, RMS_W, LN_G, LN_B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 compute, fp16 rounding of output like PyTorch)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU (fp32 compute on fp16 input, fp16 output like PyTorch)
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # softmax 2
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm (explicit fp32, cast to fp16, then fp32 mul with weight -> fp16)
    ms = tl.sum(tl.where(mask, x * x, 0.0), 0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w = tl.load(RMS_W + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 compute, eps=1e-5)
    x = tl.where(mask, x, 0.0)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(LN_G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(OUT + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback reference path
            x = x @ self.W0
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln5_g, self.ln5_b)
            return x

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        # cuBLAS/tensor-core matmul
        h = torch.matmul(x2, self.W0)
        h = h.contiguous()

        rows, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_post_kernel[(rows,)](
            h, self.rms4_w, self.ln5_g, self.ln5_b, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
