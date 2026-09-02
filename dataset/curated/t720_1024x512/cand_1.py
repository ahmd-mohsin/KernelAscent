import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 720
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_relu_scale(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load row (bf16) -> fp32
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = x * inv
    # round to bf16 (matches .to(x.dtype)), then multiply by weight in fp32 opmath, round to bf16
    xn_bf = xn.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xn_bf.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax: fp32 accumulation, output rounded to bf16
    hf = h.to(tl.float32)
    hf_m = tl.where(mask, hf, float("-inf"))
    mx = tl.max(hf_m, axis=0)
    e = tl.exp(hf_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # relu(relu(x)) is identity for softmax outputs (>=0); scale in fp32 opmath, round bf16
    out = (sm.to(tl.float32) * 1.1833).to(tl.bfloat16)
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_rms_softmax_relu_scale[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
