import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 777
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, g_ptr, b_ptr, b3_ptr, out_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x + b0, rounded to bf16 (matches reference intermediate dtype)
    a = (x + b0).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch internal accumulation)
    mean = tl.sum(a, axis=0) / N
    diff = tl.where(mask, a - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (a - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * y * (1 + erf(y / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    z = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    z = z.to(tl.bfloat16).to(tl.float32)

    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z + b3).to(tl.bfloat16)
    tl.store(out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.b3, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
