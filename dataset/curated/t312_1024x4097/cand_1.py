import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 312
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_ln_rms_kernel(
    Y, G, B, W, O,
    N, stride,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's fp16 layer_norm behavior)
    mean = tl.sum(y, axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    xhat = d * tl.math.rsqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = (xhat * g + b).to(tl.float16)  # layer_norm output rounded to fp16

    # x * 1.0028 (fp32 opmath, rounded back to fp16, matching PyTorch)
    x2 = (x1.to(tl.float32) * scale).to(tl.float16)

    # RMSNorm in fp32, round to fp16, then multiply by weight (fp32 opmath)
    xf = x2.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = (xf * tl.math.rsqrt(ms + eps_rms)).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    o = (r.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(O + row * stride + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(N)  # 512
        _fused_ln_rms_kernel[(rows,)](
            y, self.ln1_g, self.ln1_b, self.rms3_w, out,
            N, y.stride(0),
            1e-5, 1e-6, 1.0028,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
