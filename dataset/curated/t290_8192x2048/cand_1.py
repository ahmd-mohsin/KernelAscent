import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 290
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    n_cols,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=float('-inf'))
    # emulate fp16 multiply (round to fp16 after scaling), then softmax in fp32
    x = (x.to(tl.float32) * scale).to(tl.float16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    # relu is a no-op on softmax outputs (all >= 0)
    tl.store(Y + row * n_cols + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores, fp16)
        h = x @ self.W0
        h = h.contiguous()
        n_rows, n_cols = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n_cols)
        _scale_softmax_kernel[(n_rows,)](
            h, out,
            n_cols,
            1.4148,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
