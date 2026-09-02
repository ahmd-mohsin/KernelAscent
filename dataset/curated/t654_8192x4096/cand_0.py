import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 654
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_norm_act_kernel(
    X, W_rms, G, B, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, multiplied by fp16 weight in fp16)
    ms = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    x_rms = (x * inv_rms).to(tl.float16)
    w = tl.load(W_rms + cols, mask=mask, other=0.0)
    y = (x_rms * w).to(tl.float32)

    # LayerNorm (fp32 internal math, like PyTorch on half inputs)
    y_m = tl.where(mask, y, 0.0)
    mean = tl.sum(y_m, axis=0) / N
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mean) * rstd) * g + b
    z = z.to(tl.float16)

    # ReLU (half)
    z = tl.maximum(z, 0.0)

    # GELU (erf-based, fp32 opmath, cast to half)
    zf = z.to(tl.float32)
    ge = zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    ge = ge.to(tl.float16)

    # scale (opmath fp32, cast half)
    out = (ge.to(tl.float32) * 1.3529).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_act_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            n, x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
