import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 486
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_softmax(
    X, B1, B2, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf) in fp32, then round to fp16 to match reference dtype behavior
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g16 = g.to(tl.float16)

    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # bias adds in fp16 (matching reference rounding)
    h = (g16 + b1) + b2

    # softmax with fp32 accumulation (matching PyTorch half softmax)
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    row_max = tl.max(hf, axis=0)
    e = tl.exp(hf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_gelu_bias_softmax[(Mrows,)](
            x, self.b1, self.b2, y,
            x.stride(0), y.stride(0),
            N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
