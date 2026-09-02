import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 526
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X, Y,
    W1, W2, G3, B3, G4, B4,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- load row (fp32 compute) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 internal, round to bf16 like torch.softmax on bf16) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(x * x, 0) / N
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms = tl.sum(x * x, 0) / N
    x = x * tl.math.rsqrt(ms + 1e-6)
    x = x.to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 4 ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    x = d * inv * g4 + b4

    tl.store(Y + row * stride_y + cols, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xx = torch.softmax(x, dim=-1)
            _xf = xx.float(); xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms1_w
            _xf = xx.float(); xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms2_w
            xx = F.layer_norm(xx, (xx.shape[-1],), self.ln3_g, self.ln3_b)
            xx = F.layer_norm(xx, (xx.shape[-1],), self.ln4_g, self.ln4_b)
            return xx

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_row_kernel[(rows,)](
            x2d, y,
            self.rms1_w, self.rms2_w,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
