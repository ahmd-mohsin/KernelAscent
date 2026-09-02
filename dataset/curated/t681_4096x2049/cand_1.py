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
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add: bf16 + bf16 computed in fp32, rounded back to bf16 (match PyTorch)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32 (PyTorch upcasts bf16 to fp32 internally)
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # softmax output rounded to bf16, relu(relu(x)) is identity (softmax >= 0),
    # then scale in fp32 and round back to bf16 (match PyTorch scalar mul)
    out = (sm.to(tl.bfloat16).to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 tensor-core matmul (fp32 accumulate)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _bias_softmax_scale_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            1.1457,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
