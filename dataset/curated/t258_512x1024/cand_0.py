import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 258
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    # Emulate PyTorch bf16 elementwise semantics: compute in fp32, round to bf16 per op
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.2924).to(tl.bfloat16)
    v = (v.to(tl.float32) * 1.4915).to(tl.bfloat16)

    f = v.to(tl.float32)
    f = tl.where(mask, f, float('-inf'))
    m = tl.max(f, 0)
    e = tl.exp(f - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (identical to reference matmul)
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_scale_softmax_kernel[(Mrows,)](
            y, self.b1, out,
            y.stride(0), out.stride(0),
            N, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
