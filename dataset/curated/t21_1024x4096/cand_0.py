import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 21
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    x_ptr, b_ptr, out_ptr,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = x_ptr + row * stride_row + offs

    # ---- load input (bf16 -> fp32) ----
    x = tl.load(ptr).to(tl.float32)

    # ---- gelu (exact, erf-based) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    # round to bf16 to match op-boundary precision of reference
    g = g.to(tl.bfloat16).to(tl.float32)

    # ---- softmax (fp32 accumulation, like PyTorch) ----
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- gelu ----
    g2 = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    g2 = g2.to(tl.bfloat16)

    # ---- add bias (bf16 add, like reference) ----
    b = tl.load(b_ptr + offs)
    z = (g2 + b).to(tl.float32)

    # ---- softmax ----
    m2 = tl.max(z, axis=0)
    e2 = tl.exp(z - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.shape[-1] == 4096 and x.dtype == torch.bfloat16:
            orig_shape = x.shape
            x2 = x.contiguous().view(-1, 4096)
            n_rows = x2.shape[0]
            out = torch.empty_like(x2)
            b = self.b3
            if b.device != x2.device:
                b = b.to(x2.device)
            _fused_row_kernel[(n_rows,)](
                x2, b, out,
                x2.stride(0),
                BLOCK=4096,
                num_warps=8,
            )
            return out.view(orig_shape)
        # fallback (reference path)
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        x = F.gelu(x)
        x = x + self.b3
        x = torch.softmax(x, dim=-1)
        return x
