import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 61
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _scale_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)
    b = tl.load(B_ptr + cols).to(tl.float32)

    # Emulate bf16 rounding of the reference elementwise ops:
    # x = (x * 1.1673)  -> bf16 result
    t = (x * SCALE).to(tl.bfloat16).to(tl.float32)
    # x = x + b4        -> bf16 result
    u = (t + b).to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    m = tl.max(u, axis=0)
    e = tl.exp(u - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmuls with in-place ReLU between them
        h = torch.mm(x, self.W0)
        h.relu_()
        z = torch.mm(h, self.W2)

        rows, cols = z.shape
        out = torch.empty_like(z)
        _scale_bias_softmax_kernel[(rows,)](
            z, self.b4, out,
            z.stride(0), out.stride(0),
            SCALE=1.1673,
            BLOCK=cols,
            num_warps=8,
        )
        return out
