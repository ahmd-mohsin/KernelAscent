import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 759
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _softmax_bias_relu_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load row (bf16 -> fp32), matching PyTorch's float accumulation for softmax
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p = e / s

    # PyTorch softmax writes bf16 output, then the add upcasts to fp32 again
    p_bf16 = p.to(tl.bfloat16)
    p_f32 = p_bf16.to(tl.float32)

    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = p_f32 + b
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS (bf16 in, fp32 accumulate) - identical to reference
        z = x @ self.W0

        orig_shape = z.shape
        z2 = z.reshape(-1, orig_shape[-1])
        if not z2.is_contiguous():
            z2 = z2.contiguous()

        Mrows, N = z2.shape
        y = torch.empty_like(z2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_bias_relu_kernel[(Mrows,)](
            z2, self.b2, y,
            N, z2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
