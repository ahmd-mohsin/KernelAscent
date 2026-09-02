import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 222
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _gelu_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # Exact GELU (erf-based), computed in fp32 (matches PyTorch opmath for half)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # Round to fp16 (gelu output dtype), then promote for RMS stats (matches _xf = x.float())
    gh = g.to(tl.float16).to(tl.float32)

    ms = tl.sum(gh * gh, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + eps)

    # (normalized).to(fp16) then multiply by weight in fp32 opmath, round to fp16
    norm_h = (gh * rinv).to(tl.float16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (norm_h * w).to(tl.float16)

    tl.store(Y_ptr + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _gelu_rmsnorm_kernel[(rows,)](
            h, self.rms2_w, y,
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
