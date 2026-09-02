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
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # relu (bf16, no rounding concerns)
    x = tl.maximum(x, 0.0)

    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)

    # Match PyTorch elementwise semantics: compute each add in fp32,
    # then round back to bf16 after each op.
    x = (x.to(tl.float32) + b2.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b3.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)

    # Softmax in fp32 (matches PyTorch's accumulation type for bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Out_ptr + row * stride_o + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same op as reference)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_relu_bias_softmax[(M_,)](
            h, self.b2, self.b3, self.b4, out,
            h.stride(0), out.stride(0),
            N=N_, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
