import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 258
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X, B, OUT,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # bf16

    # replicate bf16 rounding at each elementwise step
    t = (x + b).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.2924).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.4915).to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch bf16 softmax which upcasts)
    v = t.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _bias_scale_softmax_kernel[(m,)](
            y, self.b1, out,
            y.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
