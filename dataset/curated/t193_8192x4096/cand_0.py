import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 193
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm (mean over N valid elements)
    ms = tl.sum(xf * xf, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    # match reference: (xf * rsqrt).to(fp16) * w  (fp16 multiply)
    v16 = (xf * rs).to(tl.float16) * w
    v = v16.to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation)
    v = tl.where(mask, v, float('-inf'))
    vmax = tl.max(v, axis=0)
    e = tl.exp(v - vmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x * 1.0122
        y = x @ self.W1  # cuBLAS fp16 tensor-core GEMM
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty((m, n), dtype=torch.float16, device=y.device)
        BLOCK = triton.next_power_of_2(n)
        _rms_softmax_kernel[(m,)](
            y, self.rms2_w, out,
            y.stride(0),
            N=n, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
