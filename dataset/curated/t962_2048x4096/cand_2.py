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
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptrs = X_ptr + row * stride_row + offs

    # Load matmul output (bf16) and compute exact (erf) GELU in fp32
    x = tl.load(x_ptrs).to(tl.float32)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # Round to bf16 to match PyTorch intermediate storage
    g = g.to(tl.bfloat16)

    # Bias add (fp32 opmath, round back to bf16 like PyTorch)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = (g.to(tl.float32) + b).to(tl.bfloat16)

    # ReLU
    y = tl.maximum(y, tl.zeros_like(y))

    # Softmax in fp32 (matches PyTorch acc type for bf16 softmax)
    yf = y.to(tl.float32)
    row_max = tl.max(yf, axis=0)
    num = tl.exp(yf - row_max)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out_ptr + row * stride_row + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core matmul (cuBLAS)
        h = x @ self.W0  # (M, 2048) bf16
        h = h.contiguous()
        out = torch.empty_like(h)
        rows, cols = h.shape
        BLOCK = triton.next_power_of_2(cols)  # 2048, whole row fits in one block
        _fused_gelu_bias_relu_softmax[(rows,)](
            h, self.b2, out,
            h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
