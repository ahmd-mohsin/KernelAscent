import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 25
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (compute in fp32, round to bf16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    a = (xf * r).to(tl.bfloat16)

    # * rms1_w (bf16 * bf16 -> computed in fp32, output bf16, matching PyTorch opmath)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = (a.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # scalar multiplies (each op rounds to bf16 like separate PyTorch ops)
    c = (b.to(tl.float32) * 1.2879).to(tl.bfloat16)
    d = (c.to(tl.float32) * 1.2562).to(tl.bfloat16)

    # softmax in fp32, output bf16 (matching PyTorch bf16 softmax)
    df = d.to(tl.float32)
    df = tl.where(mask, df, float("-inf"))
    mval = tl.max(df, axis=0)
    e = tl.exp(df - mval)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # final scale
    out = (sm.to(tl.float32) * 1.2744).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_softmax_kernel[(m,)](
            x, self.rms1_w, out,
            n, x.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
