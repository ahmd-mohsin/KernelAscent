import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 42
M, D, DT = 512, 512, torch.float16


@triton.jit
def softmax_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm, stride_om,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # cast to fp16 and back (match reference: softmax output is fp16, then .float())
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    # rmsnorm
    ms = tl.sum(pf * pf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    y = (pf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = y * w
    out = out * SCALE

    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        softmax_rms_kernel[(m,)](
            x, self.rms2_w, out,
            x.stride(0), out.stride(0),
            N=n,
            SCALE=1.1093,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
