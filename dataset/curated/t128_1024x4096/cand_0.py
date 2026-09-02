import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_chain_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 compute, fp16 output like torch) ----
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    x = e / tl.sum(e, 0)
    x = x.to(tl.float16).to(tl.float32)

    # ---- layer_norm (fp32 compute, fp16 output) ----
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g + b
    x = x.to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    x = tl.where(mask, x, float('-inf'))
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    x = e / tl.sum(e, 0)
    x = x.to(tl.float16).to(tl.float32)

    # ---- rmsnorm: fp32 norm, cast to fp16, then fp16 * fp16 weight ----
    xm = tl.where(mask, x, 0.0)
    ms = tl.sum(xm * xm, 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)  # fp16
    x = (xh * w).to(tl.float32)

    # ---- softmax 3 ----
    x = tl.where(mask, x, float('-inf'))
    x = x - tl.max(x, 0)
    e = tl.exp(x)
    x = e / tl.sum(e, 0)

    tl.store(Out_ptr + row * stride_row + offs, x.to(tl.float16), mask=mask)


SEED = 128
M, D, DT = 1024, 4096, torch.float16


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_chain_kernel[(rows,)](
            y, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
