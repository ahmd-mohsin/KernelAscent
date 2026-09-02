import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 806
M, D, DT = 4096, 512, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, b3_ptr, out_ptr,
                  n_rows, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(b3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x + b0 (round to bf16 to match reference elementwise semantics)
    v = (x + b0).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * v * (1 + erf(v / sqrt(2)))
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # + b3
    s = (g + b3).to(tl.bfloat16).to(tl.float32)

    # softmax along the row (fp32 accumulation, matching PyTorch's upcast)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            v = x + self.b0
            v = F.gelu(v)
            v = torch.relu(v)
            v = v + self.b3
            return torch.softmax(v, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(n_rows,)](
            x2, self.b0, self.b3, out,
            n_rows, D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
