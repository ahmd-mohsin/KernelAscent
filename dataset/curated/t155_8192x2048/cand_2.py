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
    X, G, B, B2, W, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row, relu (exact in any dtype)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # LayerNorm in fp32 (matches PyTorch bf16 layer_norm compute)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to bf16 like layer_norm output
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b2 in bf16 (fp32 compute, bf16 rounding)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2)
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    z = (y * r).to(tl.bfloat16).to(tl.float32)

    # * rms weight (bf16 rounding)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.bfloat16).to(tl.float32)

    # * scalar (bf16 rounding)
    z = (z * 1.0873).to(tl.bfloat16)

    tl.store(Y + row * N + offs, z, mask=mask)


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
            # CPU fallback: reference path
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = x * 1.0873
            return x

        x = x.contiguous()
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2d, self.ln1_g, self.ln1_b, self.b2, self.rms3_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8, num_stages=1,
        )
        return out.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
