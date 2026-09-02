import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 82
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # fused ReLU
    x = tl.maximum(x, 0.0)
    # mask out-of-range lanes so they don't contribute to softmax
    x = tl.where(mask, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs,
             y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # scale (matches reference elementwise op exactly), then cuBLAS GEMM
        x = x * 1.3815
        h = x @ self.W1  # (M, 512) bf16, tensor-core GEMM

        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _relu_softmax_kernel[(M_,)](
            h, out,
            N_, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
