import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 990
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_ln_relu_rms_kernel(
    X, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * g + b

    # cast to bf16 (layer_norm output dtype), then relu
    y = y.to(tl.bfloat16)
    zero = tl.zeros_like(y)
    y = tl.maximum(y, zero)

    # RMSNorm in fp32 on the bf16-rounded values
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yf * r).to(tl.bfloat16).to(tl.float32) * w
    out = out.to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = torch.relu(y)
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_relu_rms_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            N=N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
