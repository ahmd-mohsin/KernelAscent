import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 528
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, B3, W, OUT,
    N, stride_x, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch's bf16 path)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    # round to bf16 (output of layer_norm), then next op in fp32 opmath
    y = y.to(tl.bfloat16).to(tl.float32)

    # * 1.3076 (fp32 opmath, round to bf16)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # relu (exact)
    y = tl.maximum(y, 0.0)

    # + b3 (fp32 opmath, round to bf16)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    y_m = tl.where(mask, y, 0.0)
    ms = tl.sum(y_m * y_m, axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)

    yn = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yn * w).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y * 1.3076
            y = torch.relu(y)
            y = y + self.b3
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.b3, self.rms4_w, out,
            N, x2.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6, SCALE=1.3076,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
