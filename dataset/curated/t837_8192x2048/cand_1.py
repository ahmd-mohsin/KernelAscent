import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 837
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W1_ptr, W2_ptr, Out_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offs = row * N + cols

    # ---- first RMSNorm ----
    x = tl.load(X_ptr + offs).to(tl.float32)
    ms1 = tl.sum(x * x, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + eps)
    # normalize, round to bf16 (matches .to(x.dtype))
    y_bf = (x * r1).to(tl.bfloat16)
    # bf16 * bf16 elementwise (fp32 compute, round back to bf16)
    w1 = tl.load(W1_ptr + cols).to(tl.float32)
    x1_bf = (y_bf.to(tl.float32) * w1).to(tl.bfloat16)

    # ---- second RMSNorm ----
    x1 = x1_bf.to(tl.float32)
    ms2 = tl.sum(x1 * x1, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + eps)
    z_bf = (x1 * r2).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + cols).to(tl.float32)
    out_bf = (z_bf.to(tl.float32) * w2).to(tl.bfloat16)

    tl.store(Out_ptr + offs, out_bf)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, out,
            N, 1e-6,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out @ self.W3
