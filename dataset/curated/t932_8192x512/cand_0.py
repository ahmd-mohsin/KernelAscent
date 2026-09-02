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
    X, W, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale by 1.069 (fp32 opmath, round to bf16 like PyTorch)
    xf = x.to(tl.float32) * 1.069
    xb = xf.to(tl.bfloat16)

    # exact GELU (erf) in fp32 opmath, rounded to bf16
    xf = xb.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    gb = g.to(tl.bfloat16)

    # RMSNorm in fp32 on the bf16-rounded gelu output
    gf = gb.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    nb = (gf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (nb.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core bf16 matmul
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_scale_gelu_rms_kernel[(M_,)](
            x, self.rms3_w, out,
            x.stride(0), out.stride(0),
            N=N_, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
