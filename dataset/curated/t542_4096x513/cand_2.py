import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 542
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_softmax_bias_ln_kernel(
    X_ptr, OUT_ptr, B2_ptr, G_ptr, Bt_ptr,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32
    row_max = tl.max(x, axis=0)
    x = x - row_max
    ex = tl.exp(x)
    denom = tl.sum(tl.where(mask, ex, 0.0), axis=0)
    sm = ex / denom

    # round to fp16 (matches reference storing softmax output in fp16)
    sm_h = sm.to(tl.float16)

    # add bias in fp16 (matches reference fp16 add)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0)
    y_h = sm_h + b2

    # layernorm in fp32
    y = y_h.to(tl.float32)
    y = tl.where(mask, y, 0.0)
    mean = tl.sum(y, axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bt_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y - mean) * rstd * g + b

    tl.store(OUT_ptr + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8
        _fused_softmax_bias_ln_kernel[(m,)](
            h, out, self.b2, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
