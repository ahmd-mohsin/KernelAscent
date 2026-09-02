import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 665
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    # round to bf16 (matches intermediate storage in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm on the bf16-rounded values (eps=1e-6)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)

    z = (y * rrms).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.bfloat16).to(tl.float32)
    z = (z * 1.451).to(tl.bfloat16)

    tl.store(Y + row * N + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        h = h.contiguous()
        rows, n = h.shape
        out = torch.empty_like(h)
        _fused_ln_rms_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
