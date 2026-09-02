import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 655
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_scale_ln_rms(
    X, G, B, W, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (fp16), promote to fp32
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # scale: PyTorch computes in fp32 then rounds to fp16
    x = (x * SCALE).to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch half layer_norm accumulation)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b

    # round to fp16 (layer_norm output dtype), then RMSNorm in fp32
    y = y.to(tl.float16).to(tl.float32)
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)

    # (_xf * rsqrt).to(fp16) * rms3_w  -> final multiply happens in fp16
    yh = (y * rrms).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16
    out = yh * w

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (fp16 in / fp32 accumulate), same as reference
        h = torch.matmul(x, self.W0)

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_ln_rms[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.4538,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
