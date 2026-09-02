import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 699
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)

    # replicate: (x * 1.3481) in fp32 -> round to fp16 -> (* 1.262) in fp32 -> round to fp16
    v = x.to(tl.float32) * 1.3481
    v = v.to(tl.float16)
    v = v.to(tl.float32) * 1.262
    v = v.to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    v32 = tl.where(mask, v.to(tl.float32), float('-inf'))
    m = tl.max(v32, axis=0)
    e = tl.exp(v32 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM with fp32 accumulate (same as reference matmul)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _scale_softmax_kernel[(m,)](
            h, y, n,
            h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
