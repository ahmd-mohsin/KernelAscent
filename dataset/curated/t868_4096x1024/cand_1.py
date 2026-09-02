import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 868
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    # relu (exact in fp16), then scale in fp32 and round back to fp16
    # to match the reference's intermediate fp16 tensor
    x = tl.maximum(x, 0.0)
    x = (x.to(tl.float32) * scale).to(tl.float16)

    # softmax in fp32
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_ym + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _relu_scale_softmax_kernel[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n, 1.3025,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
