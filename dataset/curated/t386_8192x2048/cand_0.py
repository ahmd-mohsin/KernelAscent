import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 386
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,      # row length (512)
    BLOCK: tl.constexpr,  # power-of-2 >= N
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x_bf = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    x = x_bf.to(tl.float32)

    # RMSNorm (computed in fp32, matching reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    n = x * inv
    # cast to bf16 (matching .to(x.dtype)), then back to fp32
    n = n.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, output rounded to bf16 like torch.softmax on bf16)
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    soft = (e / s).to(tl.bfloat16).to(tl.float32)

    # scale by 1.4014 (fp32 opmath, round to bf16)
    t = (soft * 1.4014).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based, fp32 opmath), then ReLU
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    g = tl.maximum(g, 0.0)

    tl.store(Y_ptr + row * N + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_rms_softmax_gelu_kernel[(m,)](
            x, self.rms1_w, out,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
