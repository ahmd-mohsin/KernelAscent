import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 905
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 and back (match reference: softmax output stored in bf16, then .float())
    p_bf = p.to(tl.bfloat16)
    pf = p_bf.to(tl.float32)

    # rms norm
    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    normed = (pf * r).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = normed * w

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, y,
            N, h.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
