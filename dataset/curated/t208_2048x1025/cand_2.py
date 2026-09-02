import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 208
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # scale + relu + relu
    x = x * SCALE
    x = tl.maximum(x, 0.0)
    # masked-out lanes stay -inf so they don't affect max; set to -inf after relu
    x = tl.where(mask, x, float('-inf'))

    # softmax (fp32 accumulation, matching PyTorch's internal upcast for bf16)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (bitwise identical to reference matmul)
        h = x @ self.W0

        if not h.is_cuda:
            h = torch.relu(h * 1.0675)
            h = torch.softmax(h, dim=-1)
            return torch.relu(h)

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_scale_relu_softmax[(m,)](
            h, out,
            h.stride(0), out.stride(0),
            n,
            SCALE=1.0675,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        # final relu is a no-op on softmax outputs (all >= 0)
        return out
