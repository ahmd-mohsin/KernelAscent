import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 524
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_scale_bias_relu_softmax(
    X_ptr, B_ptr, Y_ptr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # load matmul output (bf16) and upcast to fp32 for op-math
    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)

    # emulate bf16 rounding after each elementwise op (matches PyTorch semantics)
    x = (x * 1.2872).to(tl.bfloat16).to(tl.float32)
    x = (x * 1.4411).to(tl.bfloat16).to(tl.float32)

    b = tl.load(B_ptr + cols).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # relu (exact in bf16)
    x = tl.maximum(x, 0.0)

    # softmax in fp32 (matches PyTorch bf16 softmax accumulation)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (identical to reference matmul)
        h = x @ self.W0
        h = h.contiguous()

        M_rows, N = h.shape
        out = torch.empty_like(h)

        _fused_scale_bias_relu_softmax[(M_rows,)](
            h, self.b3, out,
            h.stride(0), out.stride(0),
            BLOCK=N,
            num_warps=8,
        )
        return out
