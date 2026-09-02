import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 132
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))
    # Emulate the reference fp16 scaling step (x * 1.3995 stored in fp16),
    # then compute softmax in fp32 (matching PyTorch's fp16 softmax which
    # accumulates in fp32).
    xf = x.to(tl.float32) * SCALE
    xf = xf.to(tl.float16).to(tl.float32)

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Two GEMMs (fewer FLOPs than a fused single GEMM) via cuBLAS tensor cores.
        h = torch.mm(x, self.W0)
        z = torch.mm(h, self.W1)

        if not z.is_cuda:
            z = z * 1.3995
            return torch.softmax(z, dim=-1)

        z = z.contiguous()
        rows, N = z.shape
        out = torch.empty_like(z)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _scale_softmax_kernel[(rows,)](
            z, out,
            z.stride(0), out.stride(0),
            N,
            SCALE=1.3995,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
