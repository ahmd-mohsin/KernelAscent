import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 234
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax_kernel(
    x_ptr, b2_ptr, b4_ptr, out_ptr,
    D,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4841 (compute fp32, round to bf16 like PyTorch)
    x = (x * 1.4841).to(tl.bfloat16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # exact GELU (erf-based), round to bf16 after op
    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    x = (x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b4).to(tl.bfloat16).to(tl.float32)

    # softmax over the row in fp32 (matches PyTorch bf16 softmax accumulation)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x * 1.4841
            x = F.gelu(x)
            x = x + self.b2
            x = F.gelu(x)
            x = x + self.b4
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_bias_softmax_kernel[(rows,)](
            x2, self.b2, self.b4, out,
            d,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
