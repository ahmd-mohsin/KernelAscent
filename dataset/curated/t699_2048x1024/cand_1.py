import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 699
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)

    # Replicate: (x * 1.3481) rounded to fp16, then (* 1.262) rounded to fp16
    # (PyTorch half elementwise ops compute in fp32 then round to half)
    x = (x.to(tl.float32) * 1.3481).to(tl.float16)
    x = (x.to(tl.float32) * 1.262).to(tl.float16)

    xf = tl.where(mask, x.to(tl.float32), float("-inf"))

    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (same as reference)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(Mrows,)](
            h, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
