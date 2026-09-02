import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 380
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, OUT_ptr,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    # relu (bf16 -> exact, relu doesn't round)
    x = tl.maximum(x, 0.0).to(tl.float32)

    # layernorm in fp32 (matches PyTorch mixed-precision layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # cast to bf16 (layer_norm output dtype) then relu
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    # softmax in fp32, output cast to bf16
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # rmsnorm-style scaling
    ms = tl.sum(sm * sm, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (sm * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.bfloat16)
    tl.store(OUT_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        Mr, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mr,)](
            h, self.ln2_g, self.ln2_b, self.rms5_w, out,
            N, h.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
