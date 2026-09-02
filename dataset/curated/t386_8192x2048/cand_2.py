import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 386
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16 like reference)
    ms = tl.sum(x * x, axis=0) / N
    rs = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * rs).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    xw = (xn * w).to(tl.bfloat16).to(tl.float32)

    # Softmax in fp32, result rounded to bf16 (matches PyTorch bf16 softmax)
    xw_masked = tl.where(mask, xw, float("-inf"))
    m = tl.max(xw_masked, axis=0)
    e = tl.exp(xw_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # scale (fp32 opmath, round to bf16)
    sc = (sm * 1.4014).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) in fp32, round to bf16
    g = sc * 0.5 * (1.0 + tl.math.erf(sc * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    out = tl.maximum(g, 0.0).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
