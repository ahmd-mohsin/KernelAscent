import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, W, B2, G3, B3, OUT,
    stride_x, stride_o,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (float32 accumulation), then round to fp16
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (x * r).to(tl.float16)

    # * rms0_w  (fp16 * fp16 -> fp16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (y * w).to(tl.float16)

    # * 1.0401 (computed in fp32 opmath, rounded to fp16, matching PyTorch scalar mul)
    y = (y.to(tl.float32) * 1.0401).to(tl.float16)

    # + b2 (fp16 add)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    y = (y + b2).to(tl.float16)

    # LayerNorm in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / D
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * inv * g + b)

    # ReLU
    out = tl.maximum(out, 0.0).to(tl.float16)
    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = y * 1.0401
            y = y + self.b2
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(y)

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_kernel[(m,)](
            x2, self.rms0_w, self.b2, self.ln3_g, self.ln3_b, out,
            x2.stride(0), out.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
