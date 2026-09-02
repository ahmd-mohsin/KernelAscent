import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 362
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _softmax_scale_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # softmax 1 (fp32 compute, round result to bf16 like PyTorch out dtype)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # scale (fp32 opmath, round to bf16)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # add bias (fp32 opmath, round to bf16)
    b = tl.load(B_ptr + offs).to(tl.float32)
    y = (y + b).to(tl.bfloat16).to(tl.float32)

    # softmax 2
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, cols = h.shape

        out = torch.empty_like(h)
        grid = (rows,)
        _softmax_scale_bias_softmax_kernel[grid](
            h, self.b3, out,
            h.stride(0), out.stride(0),
            SCALE=1.2039,
            BLOCK=cols,
            num_warps=8,
        )
        return out
