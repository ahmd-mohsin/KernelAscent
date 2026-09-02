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
    X, Y,
    N, stride_xm, stride_ym,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # replicate fp16 relu + fp16 scale exactly as reference
    x = tl.maximum(x, 0.0)
    x = x * scale.to(x.dtype)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for half inputs)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.3025
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _relu_scale_softmax_kernel[(Mrows,)](
            x, y, N, x.stride(0), y.stride(0), 1.3025,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
