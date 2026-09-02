import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 993
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _softmax_scale_bias_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # match reference: (softmax.bf16 * scale).bf16 + bias
    sm_bf = sm.to(tl.bfloat16)
    y = (sm_bf.to(tl.float32) * SCALE).to(tl.bfloat16)
    out = (y.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # tensor-core matmul
        if not z.is_cuda:
            z = torch.softmax(z, dim=-1)
            z = z * 1.1294
            z = z + self.b3
            return z
        z = z.contiguous()
        Mrows, N = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_scale_bias_kernel[(Mrows,)](
            z, self.b3, out,
            N, z.stride(0), out.stride(0),
            SCALE=1.1294,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
