import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 572
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, w2_ptr, g3_ptr, b3_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * D

    x = tl.load(x_ptr + base + offs).to(tl.float32)
    b0 = tl.load(b0_ptr + offs).to(tl.float32)

    # x = relu(x + b0)  (bf16 add semantics: emulate by rounding to bf16)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # RMSNorm in fp32, cast to bf16, multiply by w2 in bf16
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)
    w2 = tl.load(w2_ptr + offs)
    y = (xn * w2).to(tl.float32)  # bf16 mul then upcast for layernorm

    # emulate bf16 multiply rounding
    y = (xn * w2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(y, axis=0) / D
    yc = y - mean
    var = tl.sum(yc * yc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g3 = tl.load(g3_ptr + offs).to(tl.float32)
    b3 = tl.load(b3_ptr + offs).to(tl.float32)
    out = yc * rstd * g3 + b3

    tl.store(out_ptr + base + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.b0, self.rms2_w, self.ln3_g, self.ln3_b, out,
            D=d, BLOCK=d, num_warps=8,
        )
        return out
