import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 989
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, g3_ptr, b3_ptr, w4_ptr, out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4347  (bf16 op, fp32 opmath, round to bf16)
    x = (x * 1.4347).to(tl.bfloat16).to(tl.float32)

    # x = x + b1
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), fp32 opmath, round to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 internals, affine in fp32, output bf16)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gamma = tl.load(g3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(b3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((x - mean) * rstd * gamma + beta).to(tl.bfloat16).to(tl.float32)

    # RMSNorm as in reference: fp32 mean-of-squares, cast to bf16, then bf16 mul
    ym = tl.where(mask, y, 0.0)
    ms = tl.sum(ym * ym, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    z = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(w4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.4347
            x = x + self.b1
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
