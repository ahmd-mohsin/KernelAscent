import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 769
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _scale_bias_softmax_kernel(
    Y_ptr, B_ptr, Out_ptr,
    N,
    stride_ym, stride_om,
    S1, S2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_ym + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # Emulate fp16 elementwise arithmetic (round after each op) to match reference
    t = (y.to(tl.float32) * S1).to(tl.float16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    t = (t.to(tl.float32) * S2).to(tl.float16)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    f = t.to(tl.float32)
    f = tl.where(mask, f, float("-inf"))
    m = tl.max(f, axis=0)
    e = tl.exp(f - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 matmul (same as reference)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _scale_bias_softmax_kernel[(Mrows,)](
            y, self.b2, out,
            N,
            y.stride(0), out.stride(0),
            1.2814, 1.0299,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
