import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 326
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _epilogue_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    v = tl.maximum(x, 0.0)
    # x * 1.0573 (computed in fp32, rounded to bf16 like PyTorch elementwise)
    v = (v * 1.0573).to(tl.bfloat16).to(tl.float32)
    # x * 1.4239
    v = (v * 1.4239).to(tl.bfloat16).to(tl.float32)
    # second relu (kept for exactness)
    v = tl.maximum(v, 0.0)
    # + b5 (fp32 compute, bf16 round, like PyTorch)
    v = (v + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = torch.relu(y)
            y = y * 1.0573
            y = y * 1.4239
            y = torch.relu(y)
            y = y + self.b5
            return torch.softmax(y, dim=-1)

        # cuBLAS matmul (tensor cores on A100)
        y = torch.matmul(x, self.W0)

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _epilogue_softmax_kernel[(m,)](
            y, self.b5, out,
            n,
            y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
