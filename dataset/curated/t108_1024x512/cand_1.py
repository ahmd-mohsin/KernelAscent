import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 108
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_scale_rms_ln_kernel(
    X_ptr, RMSW_ptr, G_ptr, B_ptr, OUT_ptr,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)  # bf16
    xf = x.to(tl.float32)

    # x = x * 1.101 (round to bf16), then * 1.056 (round to bf16)
    xf = (xf * 1.101).to(tl.bfloat16).to(tl.float32)
    xf = (xf * 1.056).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (computed in fp32, cast to bf16, then multiply by weight in bf16)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    t = (xf * r).to(tl.bfloat16)

    w = tl.load(RMSW_ptr + cols, mask=mask, other=0.0)  # bf16
    t = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # LayerNorm in fp32
    tf = t.to(tl.float32)
    mean = tl.sum(tl.where(mask, tf, 0.0), axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    invstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = (tf - mean) * invstd * g + b
    tl.store(OUT_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # tensor-core bf16 matmul
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_rms_ln_kernel[(rows,)](
            y, self.rms3_w, self.ln4_g, self.ln4_b, out,
            y.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
