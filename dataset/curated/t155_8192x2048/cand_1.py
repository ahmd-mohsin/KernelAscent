import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 155
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, G1, B1, B2, W3,
    N, stride_x, stride_y,
    ln_eps, rms_eps, final_scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # layer norm (fp32 math, biased variance, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + ln_eps)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    # cast to bf16 (layer_norm output dtype), then back to fp32 for the add
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b2 (bf16 add computed in fp32 opmath, result bf16)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # rms norm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + rms_eps)
    y = (y * inv).to(tl.bfloat16).to(tl.float32)

    # * rms3_w (bf16*bf16 in fp32 opmath -> bf16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w3).to(tl.bfloat16).to(tl.float32)

    # * 1.0873 scalar (fp32 opmath -> bf16)
    y = (y * final_scale).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = x * 1.0873
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2d, y,
            self.ln1_g, self.ln1_b, self.b2, self.rms3_w,
            N, x2d.stride(0), y.stride(0),
            1e-5, 1e-6, 1.0873,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
