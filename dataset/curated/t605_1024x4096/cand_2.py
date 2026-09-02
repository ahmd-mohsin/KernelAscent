import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 605
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row (fp16 -> fp32), softmax computed in fp32 (matches PyTorch semantics)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.math.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm16 = (e / denom).to(tl.float16)  # softmax output rounded to fp16, as PyTorch returns

    # RMSNorm on the fp16-rounded softmax values, computed in fp32
    xf = sm16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)
    y16 = (xf * r).to(tl.float16)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    out = y16 * w  # fp16 * fp16, matching PyTorch elementwise behavior
    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
