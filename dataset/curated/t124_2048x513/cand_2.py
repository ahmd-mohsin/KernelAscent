import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 124
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_relu_bias_softmax_scale(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # relu(relu(x)) == relu(x)
    t = tl.maximum(x, 0.0)
    # x + b3 is computed in fp32 opmath then stored as fp16 in the reference
    s = (t + b).to(tl.float16).to(tl.float32)
    s = tl.where(mask, s, float("-inf"))

    # softmax in fp32 (PyTorch upcasts fp16 softmax internally)
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    r = e / denom

    # softmax output rounds to fp16, then scalar mul in fp32 opmath, round to fp16
    r_h = r.to(tl.float16).to(tl.float32)
    out = (r_h * SCALE).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_bias_softmax_scale[(Mrows,)](
            h, self.b3, y,
            N, h.stride(0), y.stride(0),
            SCALE=1.1092,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
