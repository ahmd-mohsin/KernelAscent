import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 409
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _softmax_scale_kernel(
    X, Y,
    stride_x, stride_y,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # match reference: softmax -> bf16, then scalar mul in fp32 -> bf16
    y = y.to(tl.bfloat16).to(tl.float32) * scale
    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores on A100)
        z = torch.matmul(x, self.W0)
        z = z.contiguous()

        rows, N = z.shape
        out = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(N)
        _softmax_scale_kernel[(rows,)](
            z, out,
            z.stride(0), out.stride(0),
            N, 1.1318,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
