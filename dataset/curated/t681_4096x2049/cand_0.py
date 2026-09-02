import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 681
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_scale_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (bf16) and bias, upcast to fp32 (matches PyTorch opmath)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add computed in fp32, result rounded to bf16 (as PyTorch add does),
    # then re-read as fp32 for the softmax (as PyTorch softmax upcasts bf16 input)
    t = (x + b).to(tl.bfloat16).to(tl.float32)
    t = tl.where(mask, t, float('-inf'))

    # numerically-stable softmax in fp32
    m = tl.max(t, axis=0)
    e = tl.math.exp(t - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # softmax output is written back as bf16 in the reference; relu is a no-op
    # (softmax >= 0). Scale then happens in fp32 opmath and rounds to bf16.
    sm_bf16 = sm.to(tl.bfloat16).to(tl.float32)
    y = sm_bf16 * SCALE

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (identical to reference)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _bias_softmax_scale_kernel[(Mrows,)](
            h, self.b1, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.1457,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
