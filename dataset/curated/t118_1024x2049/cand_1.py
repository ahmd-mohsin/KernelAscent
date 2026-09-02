import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 118
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_bias_double_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load input and bias (bf16), add in bf16 to match reference elementwise add
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    xb = (x + b).to(tl.bfloat16)

    # ---- first softmax (fp32 accumulation, bf16 output like PyTorch) ----
    xf = tl.where(mask, xb.to(tl.float32), float('-inf'))
    m1 = tl.max(xf, 0)
    e1 = tl.exp(xf - m1)          # masked lanes: exp(-inf)=0
    s1 = tl.sum(e1, 0)
    y1 = (e1 / s1).to(tl.bfloat16)

    # relu (softmax outputs are >= 0, but keep for exact semantics)
    y1 = tl.maximum(y1, tl.zeros_like(y1))

    # ---- second softmax ----
    yf = tl.where(mask, y1.to(tl.float32), float('-inf'))
    m2 = tl.max(yf, 0)
    e2 = tl.exp(yf - m2)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        d = self.b0.shape[0]
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_bias_double_softmax_kernel[(rows,)](
            x2, self.b0, y,
            d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
