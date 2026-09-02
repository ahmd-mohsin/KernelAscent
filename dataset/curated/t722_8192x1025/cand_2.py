import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 722
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_softmax2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # ---- load row, upcast to fp32 ----
    x = tl.load(X_ptr + base + offs).to(tl.float32)

    # ---- softmax #1 (fp32 accumulate, round to bf16 like PyTorch) ----
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- softmax #2 ----
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    x = e / tl.sum(e, axis=0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32), then bf16 * bf16 weight ----
    ms = tl.sum(x * x, axis=0) / N
    x = x * (1.0 / tl.sqrt(ms + 1e-6))
    xb = x.to(tl.bfloat16)

    w = tl.load(W_ptr + offs)
    y = xb * w

    tl.store(Y_ptr + base + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 1024), bf16

        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        _fused_softmax2_rms_kernel[(rows,)](
            h, self.rms3_w, out,
            N,
            BLOCK=1024,
            num_warps=8,
        )

        # matmul 2
        return out @ self.W4
