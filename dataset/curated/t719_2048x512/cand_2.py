import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 719
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf), computed in fp32, rounded to bf16 like PyTorch op boundary
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, eps=1e-5, biased variance)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x - mean) * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32)
    xm = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.exp(x - xm)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32, eps=1e-6), scale by w
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    rrms = tl.math.rsqrt(ms + 1e-6)
    x = (x * rrms).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    x = x * w
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU
    x = tl.maximum(x, 0.0)

    tl.store(Y + row * stride_y + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 512 else 4,
        )
        return y.view(orig_shape)
