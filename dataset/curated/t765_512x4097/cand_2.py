import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 765
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B0, G1, Bt1, W2, Y,
    N, stride_x, stride_y,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (fp32 math, round to bf16 like PyTorch)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bt1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g + b
    # cast to bf16 (layer_norm output dtype), then upcast for RMS step
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    y = (y * rrms).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, y,
            N, x2.stride(0), y.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
