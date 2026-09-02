import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 406
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load row (bf16 -> fp32), softmax computed in fp32 like torch
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast softmax result back to bf16 (matching torch.softmax output dtype),
    # then re-upcast to fp32 for RMS norm (matching _xf = x.float())
    p_bf = p.to(tl.bfloat16)
    pf = p_bf.to(tl.float32)

    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    # (_xf * rsqrt(...)).to(bf16)
    y_bf = (pf * r).to(tl.bfloat16)
    yf = y_bf.to(tl.float32)

    # multiply by rms2_w (bf16 * bf16 done in fp32 opmath, rounded to bf16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yf * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(rows,)](
            h, self.rms2_w, y,
            N, h.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
