import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 103
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_bias_scale_relu_softmax(
    X, B1, B3, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)

    # Replicate fp16 intermediate rounding of the reference implementation
    x = (x + b1).to(tl.float16).to(tl.float32)
    x = (x * 1.1112).to(tl.float16).to(tl.float32)
    x = (x + b3).to(tl.float16).to(tl.float32)
    x = (x * 1.3457).to(tl.float16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    x_for_max = tl.where(mask, x, float('-inf'))
    m = tl.max(x_for_max, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Out + row * stride_o + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_scale_relu_softmax[(Mrows,)](
            h, self.b1, self.b3, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
