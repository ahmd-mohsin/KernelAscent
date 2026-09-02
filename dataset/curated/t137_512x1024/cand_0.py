import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 137
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_gelu_softmax(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean of squares in fp32, matching reference)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # cast normalized value back to fp16 before applying weight (matches reference)
    y16 = (xf * inv).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y16 = y16 * w  # fp16 multiply, as in reference

    # exact GELU (erf-based), computed in fp32 then cast to fp16 (matches F.gelu on half)
    g = y16.to(tl.float32)
    g = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # softmax: upcast fp16 input to fp32 internally (matches torch.softmax on half)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, float("-inf"))
    mx = tl.max(gf, axis=0)
    e = tl.exp(gf - mx)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_gelu_softmax[(M_,)](
            h, self.rms1_w, out,
            N_, h.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
