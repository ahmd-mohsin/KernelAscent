import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 962
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_gelu_bias_relu_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf variant), computed in fp32 then rounded to bf16 (matches PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (matches PyTorch internal fp32 accumulation for bf16)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (same as reference)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_bias_relu_softmax[(rows,)](
            h, self.b2, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
