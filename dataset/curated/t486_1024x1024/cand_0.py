import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 486
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_softmax_kernel(
    x_ptr, b1_ptr, b2_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # Load input row, compute exact GELU in fp32 (matches PyTorch opmath for half)
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # Bias adds in fp16 (matches x + b1 then + b2 in half precision)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0)
    v = (g16 + b1) + b2

    # Softmax with fp32 accumulation (matches PyTorch internal fp32 for half softmax)
    v32 = v.to(tl.float32)
    v32 = tl.where(mask, v32, float('-inf'))
    m = tl.max(v32, axis=0)
    e = tl.exp(v32 - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, D_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_gelu_bias_softmax_kernel[(M_,)](
            x, self.b1, self.b2, out,
            D=D_, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
