import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 903
M, D, DT = 512, 513, torch.float16


@triton.jit
def _gelu2_relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First GELU (exact, computed in fp32, rounded back to fp16 like PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Second GELU
    g2 = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # ReLU
    r = tl.maximum(g2, 0.0)

    # Softmax (fp32 accumulation)
    r = tl.where(mask, r, float('-inf'))
    m = tl.max(r, axis=0)
    e = tl.exp(r - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100 for fp16)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.reshape(-1, N)
        if not h2.is_contiguous():
            h2 = h2.contiguous()

        out = torch.empty_like(h2)
        rows = h2.shape[0]
        BLOCK = triton.next_power_of_2(N)
        _gelu2_relu_softmax_kernel[(rows,)](
            h2, out,
            N, h2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.reshape(orig_shape)
