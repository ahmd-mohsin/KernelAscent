import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 219
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _softmax_scale_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)

    # softmax in fp32, round to fp16, then scale in fp16 (matches reference op order)
    y = (e / denom).to(tl.float16)
    scale = tl.full((1,), SCALE, dtype=tl.float16)
    y = y * scale

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Fused GEMM + bias epilogue (cuBLAS): x @ W0 + b1
        h = torch.addmm(self.b1, x, self.W0)
        # Second GEMM (cuBLAS tensor cores)
        z = h @ self.W2

        if not z.is_cuda:
            return torch.softmax(z, dim=-1) * 1.481

        z = z.contiguous()
        rows, n = z.shape
        out = torch.empty_like(z)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_scale_kernel[(rows,)](
            z, out,
            n, z.stride(0), out.stride(0),
            SCALE=1.481,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
