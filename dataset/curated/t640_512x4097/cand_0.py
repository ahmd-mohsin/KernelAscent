import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 640
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu(
    x_ptr, g_ptr, b_ptr, out_ptr,
    D, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    # round through bf16 to match the reference intermediate dtype
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 stats)
    mean = tl.sum(tl.where(mask, sm, 0.0), axis=0) / D
    diff = tl.where(mask, sm - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round through bf16 to match the reference intermediate dtype
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) gelu
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(out_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            return F.gelu(y)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4
        _fused_softmax_ln_gelu[(rows,)](
            x2, self.ln1_g, self.ln1_b, out,
            d, x2.stride(0), out.stride(0), 1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
