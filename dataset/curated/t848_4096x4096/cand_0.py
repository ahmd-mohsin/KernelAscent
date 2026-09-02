import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 848
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _bias_softmax_scale_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # match reference: (half + half) computed in fp32 opmath, rounded to fp16
    v = (x + b).to(tl.float16).to(tl.float32)
    v = tl.where(mask, v, float("-inf"))

    # softmax in fp32 accumulation (matches PyTorch half softmax), round to fp16
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16).to(tl.float32)

    # match reference: half * python-scalar computed in fp32 opmath, cast to fp16
    out = (y * SCALE).to(tl.float16)
    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = y + self.b1
            y = torch.softmax(y, dim=-1)
            return y * 1.3952

        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)

        rows, N = y.shape
        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 4096 else (8 if BLOCK >= 2048 else 4)

        _bias_softmax_scale_kernel[(rows,)](
            y, self.b1, y,
            y.stride(0), y.stride(0),
            N,
            SCALE=1.3952,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
