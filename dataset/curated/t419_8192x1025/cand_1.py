import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 419
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_bias_rms_softmax_kernel(
    x_ptr, b0_ptr, w_ptr, out_ptr,
    D, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load x (bf16) and bias (bf16); PyTorch elementwise bf16 ops compute in fp32
    # and round to bf16, so we replicate: fp32 add -> bf16 round
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 (matches _xf = x.float(); rsqrt(mean(xf^2) + eps))
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    # multiply by weight (bf16 elementwise -> fp32 compute, bf16 round)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch AccumulateType for bf16), output bf16
    y = tl.where(mask, y, float("-inf"))
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        probs = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_bias_rms_softmax_kernel[(Mrows,)](
            x, self.b0, self.rms1_w, probs,
            Dcols, x.stride(0), probs.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return probs @ self.W3
