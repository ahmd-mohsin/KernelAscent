import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 334
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _softmax_gelu_bias_scale_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's internal fp32 accumulation), round to bf16
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32 opmath, round to bf16
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # bias add (fp32 opmath, bf16 round)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # scale (fp32 opmath, bf16 round)
    y = (y * 1.0072).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.softmax(h, dim=-1)
            h = F.gelu(h)
            h = h + self.b3
            h = h * 1.0072
            return h

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_gelu_bias_scale_kernel[(Mrows,)](
            h, self.b3, out,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
