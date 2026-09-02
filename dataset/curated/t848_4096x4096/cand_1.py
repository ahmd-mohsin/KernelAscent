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
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # bias add in fp16 (matches reference elementwise add on fp16 tensors)
    z = x + b
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))

    # softmax in fp32 (matches PyTorch's fp32 accumulation for fp16 softmax)
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to fp16 (softmax output), then scale in fp32 and round back
    p16 = p.to(tl.float16)
    y = (p16.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Heavy lifting: cuBLAS fp16 GEMM (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _bias_softmax_scale_kernel[(rows,)](
            h, self.b1, out,
            N,
            SCALE=1.3952,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
