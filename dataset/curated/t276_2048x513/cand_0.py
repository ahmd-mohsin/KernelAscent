import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 276
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    Z_ptr, B_ptr, OUT_ptr,
    N, stride_z, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    z = tl.load(Z_ptr + row * stride_z + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.159  (fp32 compute, round to bf16 like PyTorch)
    t = (z * 1.159).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) in fp32, round back to bf16
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # x = x + b4 (fp32 compute, round to bf16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    y = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y, axis=0)
    num = tl.exp(y - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(OUT_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two cuBLAS GEMMs (tensor cores on A100)
        z = torch.matmul(torch.matmul(x, self.W0), self.W1)

        z = z.contiguous()
        m, n = z.shape
        out = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_bias_softmax[(m,)](
            z, self.b4, out,
            n, z.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
