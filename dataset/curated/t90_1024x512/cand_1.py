import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 90
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, b_ptr, out_ptr,
                  n_cols, eps, scale,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (fp32 accumulate, matching reference)
    ms = tl.sum(xf * xf, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    a = (xf * inv).to(tl.bfloat16)  # round to bf16 like .to(x.dtype)

    # multiply by weight: PyTorch computes bf16 mul in fp32 then rounds
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = (a.to(tl.float32) * w).to(tl.bfloat16)

    # exact GELU (erf), computed in fp32 then rounded to bf16
    bf = b.to(tl.float32)
    g = 0.5 * bf * (1.0 + tl.math.erf(bf * 0.7071067811865476))
    c = g.to(tl.bfloat16)

    # add bias (fp32 opmath, round bf16)
    bb = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    d = (c.to(tl.float32) + bb).to(tl.bfloat16)

    # scale (fp32 opmath, round bf16)
    o = (d.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(out_ptr + row * n_cols + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x, self.rms0_w, self.b2, out,
            n_cols, 1e-6, 1.0205,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
