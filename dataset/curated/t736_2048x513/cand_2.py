import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 736
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    # softmax (numerically stable, fp32 accumulation like PyTorch)
    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        if not h.is_cuda:
            h = torch.relu(h)
            return torch.softmax(h, dim=-1)

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _relu_softmax_kernel[(Mrows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
        )
        return out
