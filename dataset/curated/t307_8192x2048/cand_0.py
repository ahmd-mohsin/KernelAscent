import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 307
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_softmax_relu_rms_kernel(
    X_ptr, LN_G_ptr, LN_B_ptr, RMS_W_ptr, Y_ptr,
    stride_row,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 compute, bf16 rounded output like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(LN_G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 compute, bf16 rounded output) ----
    y_masked = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- scale + relu (bf16 rounded like PyTorch elementwise) ----
    p = (p * 1.3367).to(tl.bfloat16).to(tl.float32)
    p = tl.maximum(p, 0.0)

    # ---- RMSNorm (fp32 compute, bf16 round, then * weight in fp32, bf16 round) ----
    ms = tl.sum(p * p, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    r = (p * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RMS_W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (r * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        N = h.shape[-1]
        rows = h.numel() // N
        h2 = h.view(rows, N)
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax_relu_rms_kernel[(rows,)](
            h2, self.ln1_g, self.ln1_b, self.rms5_w, out,
            h2.stride(0),
            N=N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(h.shape)
