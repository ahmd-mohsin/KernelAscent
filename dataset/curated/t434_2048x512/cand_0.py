import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 434
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_softmax(
    X_ptr, B2_ptr, B3_ptr, B4_ptr, Out_ptr,
    stride_xm, stride_om,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)

    # relu in bf16
    x = tl.maximum(x, x * 0)
    # sequential bias adds with bf16 rounding at each step (matches reference)
    x = (x + b2).to(tl.bfloat16)
    x = (x + b3).to(tl.bfloat16)
    x = (x + b4).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch's internal upcast for bf16 softmax)
    f = x.to(tl.float32)
    f = tl.where(mask, f, float('-inf'))
    m = tl.max(f, axis=0)
    e = tl.exp(f - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_om + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        if not y.is_contiguous():
            y = y.contiguous()

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_bias_softmax[(Mrows,)](
            y, self.b2, self.b3, self.b4, out,
            y.stride(0), out.stride(0),
            N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
