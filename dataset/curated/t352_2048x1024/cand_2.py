import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 352
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_scale_ln_scale_kernel(
    X, G, B, OUT,
    N, stride_x, stride_o,
    PRE: tl.constexpr, POST: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # emulate bf16 rounding of the pre-scale (matches eager x = x * 1.106 in bf16)
    t = (x.to(tl.float32) * PRE).to(tl.bfloat16).to(tl.float32)

    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / N
    diff = tl.where(mask, t - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # round to bf16 (layer_norm output), then post-scale and round again
    y = y.to(tl.bfloat16).to(tl.float32)
    out = (y * POST).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.106
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return x * 1.0831

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        M_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 2048 else 8

        _fused_scale_ln_scale_kernel[(M_rows,)](
            x2, self.ln1_g, self.ln1_b, out,
            N, x2.stride(0), out.stride(0),
            PRE=1.106, POST=1.0831, EPS=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
