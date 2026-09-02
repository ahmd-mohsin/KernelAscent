import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 886
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_bias_double_rmsnorm(
    X_ptr, B_ptr, W2_ptr, W3_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + cols)          # bf16
    b = tl.load(B_ptr + cols)                            # bf16

    # x = x + b in bf16 (round to bf16), then upcast to fp32
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # first RMSNorm
    ms1 = tl.sum(xf * xf, axis=0) / N + 1e-6
    y = (xf * tl.math.rsqrt(ms1)).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + cols)
    z = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # second RMSNorm
    zf = z.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / N + 1e-6
    o = (zf * tl.math.rsqrt(ms2)).to(tl.bfloat16)
    w3 = tl.load(W3_ptr + cols)
    out = (o.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 tensor-core GEMM
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_bias_double_rmsnorm[(m,)](
            h, self.b1, self.rms2_w, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=n,
            num_warps=4,
        )
        return out
