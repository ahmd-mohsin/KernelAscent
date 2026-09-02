import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 932
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_scale_gelu_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X_ptr + row * N_COLS + cols, mask=mask, other=0.0)
    # scale in bf16 semantics: bf16 * scalar -> f32 compute -> round to bf16
    xf = x.to(tl.float32) * 1.069
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) in f32, round to bf16 (matches PyTorch opmath behavior)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in f32
    ms = tl.sum(g * g, axis=0) / N_COLS
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (g * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.bfloat16)
    tl.store(Out_ptr + row * N_COLS + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_scale_gelu_rms_kernel[(m,)](
            y, self.rms3_w, out,
            N_COLS=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
