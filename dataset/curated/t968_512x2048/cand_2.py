import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 968
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_relu_scale_bias_softmax(
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

    # relu (fp16 exact, no rounding needed)
    x = tl.maximum(x, 0.0)
    # x * 1.0064 : PyTorch computes in fp32 (opmath) then rounds to fp16
    x = (x * SCALE).to(tl.float16).to(tl.float32)
    # x + b : computed in fp32 then rounded to fp16
    x = (x + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for half inputs)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.0064
            x = x + self.b2
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        M_, N_ = x.shape
        b = self.b2
        if b.device != x.device:
            b = b.to(x.device)
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_scale_bias_softmax[(M_,)](
            x, b, y,
            x.stride(0), y.stride(0),
            N_,
            SCALE=1.0064,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
