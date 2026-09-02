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
    X_ptr, RMSW_ptr, G_ptr, B_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0)  # bf16

    # replicate the two separate bf16 multiplies (round after each)
    x = (x * 1.101).to(tl.bfloat16)
    x = (x * 1.056).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)  # cast to bf16 as in reference

    rmsw = tl.load(RMSW_ptr + cols, mask=mask, other=0.0)  # bf16
    x2 = (xn * rmsw).to(tl.bfloat16)  # bf16 multiply, rounded

    # LayerNorm: PyTorch upcasts to fp32 internally for bf16
    x2f = x2.to(tl.float32)
    mean = tl.sum(x2f, axis=0) / N
    diff = tl.where(mask, x2f - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x2f - mean) * rstd * g + b

    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core matmul
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_rms_ln_kernel[(m,)](
            x, self.rms3_w, self.ln4_g, self.ln4_b, y,
            n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
