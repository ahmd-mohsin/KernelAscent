import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 762
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_ln_kernel(
    X_ptr, B2_ptr, B3_ptr, G_ptr, B_ptr, Y_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, matching PyTorch's fp32 opmath for half inputs
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    h = g.to(tl.float16)

    # bias adds in fp16 (matching eager x + b2, x + b3 rounding)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0)
    h = (h + b2).to(tl.float16)
    h = (h + b3).to(tl.float16)

    # layernorm with fp32 accumulation
    xf = h.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gg = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = diff * rstd * gg + bb

    tl.store(Y_ptr + row * stride_row + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = F.gelu(h) + self.b2 + self.b3
            return F.layer_norm(h, (h.shape[-1],), self.ln4_g, self.ln4_b)

        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_bias_ln_kernel[(m,)](
            h, self.b2, self.b3, self.ln4_g, self.ln4_b, y,
            n, h.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
