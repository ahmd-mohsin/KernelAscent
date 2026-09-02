import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import triton.language.extra.libdevice as _libdevice
    _HAS_LD = True
except Exception:
    try:
        from triton.language.math import exp as _tl_exp  # noqa
        _HAS_LD = False
    except Exception:
        _HAS_LD = False

SEED = 161
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_bias_dsoftmax_rms_kernel(
    X, B1, B2, W, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_xm + cols)                 # fp16
    b1 = tl.load(B1 + cols)                                 # fp16
    b2 = tl.load(B2 + cols)                                 # fp16

    # bias adds in fp16 (matches x + b1 then + b2 in half precision)
    x = x + b1
    x = x + b2

    # ---- softmax #1 (fp32 accumulation, fp16 output) ----
    f = x.to(tl.float32)
    m = tl.max(f, axis=0)
    e = tl.exp(f - m)
    s = tl.sum(e, axis=0)
    y1 = (e / s).to(tl.float16)

    # ---- softmax #2 ----
    f2 = y1.to(tl.float32)
    m2 = tl.max(f2, axis=0)
    e2 = tl.exp(f2 - m2)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.float16)

    # ---- RMSNorm (fp32) then fp16 scale ----
    f3 = y2.to(tl.float32)
    ms = tl.sum(f3 * f3, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (f3 * r).to(tl.float16)

    w = tl.load(W + cols)                                   # fp16
    out = xn * w

    tl.store(OUT + row * stride_om + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM (same as reference)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        _fused_bias_dsoftmax_rms_kernel[(Mrows,)](
            h, self.b1, self.b2, self.rms5_w, out,
            h.stride(0), out.stride(0),
            N=N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=16,
            num_stages=1,
        )
        return out
