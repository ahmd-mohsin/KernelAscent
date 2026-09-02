import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 665
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b

    # Round to bf16 (layer_norm output dtype), then RMSNorm in fp32
    y = y.to(tl.bfloat16).to(tl.float32)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + RMS_EPS)

    z = (y * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.bfloat16).to(tl.float32)
    z = (z * SCALE).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_norm_kernel[(m,)](
            x, self.ln1_g, self.ln1_b, self.rms2_w, out,
            n, x.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, SCALE=1.451,
            BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
