import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 312
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_ln_scale_rms_kernel(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, like PyTorch's half layernorm) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to fp16 (layernorm output dtype)
    y = y.to(tl.float16).to(tl.float32)

    # ---- scale by 1.0028 (half elementwise op: fp32 opmath, fp16 result) ----
    y = y * SCALE
    y = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm in fp32 ----
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    z = y * r
    z = z.to(tl.float16).to(tl.float32)  # .to(x.dtype) cast in reference

    # ---- multiply by rms weight (half*half: fp32 opmath, fp16 result) ----
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = x @ self.W0
        if not h.is_contiguous():
            h = h.contiguous()

        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_scale_rms_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.rms3_w, y,
            N, h.stride(0), y.stride(0),
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.0028,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
