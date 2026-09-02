import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 162
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, w2_ptr, g3_ptr, bb3_ptr, b4_ptr, out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)

    # relu + bias add (in bf16 to match reference rounding)
    t = tl.maximum(x, 0.0) + b1

    # RMSNorm (fp32 internals, cast to bf16, then bf16 multiply by weight)
    tf = t.to(tl.float32)
    ms = tl.sum(tf * tf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0)
    y = (tf * r).to(tl.bfloat16) * w2

    # LayerNorm (fp32 opmath)
    yf = y.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g3 = tl.load(g3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb3 = tl.load(bb3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((yf - mean) * rstd) * g3 + bb3

    # cast to bf16, then bf16 add of b4 (matches reference rounding)
    b4 = tl.load(b4_ptr + cols, mask=mask, other=0.0)
    out = z.to(tl.bfloat16) + b4

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = x + self.b4
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.b1, self.rms2_w, self.ln3_g, self.ln3_b, self.b4, out,
            N, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
